"""Template-bound .docx builder: IEEE master template styles, native OMML math (inline and numbered
display equations in the template's 4-column matrix form), tables, figures, algorithms, references.

Inline markup in paragraph strings:
  $...$       inline LaTeX math -> OMML
  **...**     bold,   *...*  italic
"""
import copy, re, os, subprocess, tempfile, zipfile
from lxml import etree
import docx
from docx.shared import Pt, Inches, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
NS = {'w': W, 'm': M}
ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI']


def _el(tag, **attrs):
    p, t = tag.split(':')
    e = etree.Element(f'{{{NS[p]}}}{t}')
    for k, v in attrs.items():
        kp, kt = k.split('_', 1)
        e.set(f'{{{NS[kp]}}}{kt}', v)
    return e


class MathCache:
    """Batch LaTeX -> OMML conversion through pandoc."""
    def __init__(self):
        self.inline, self.display = {}, {}

    def convert(self, inl, disp):
        inl = [s for s in dict.fromkeys(inl) if s not in self.inline]
        disp = [s for s in dict.fromkeys(disp) if s not in self.display]
        if not inl and not disp:
            return
        md = []
        for s in inl:
            md.append(f'${s}$\n')
        for s in disp:
            md.append(f'$${s}$$\n')
        with tempfile.TemporaryDirectory() as td:
            open(f'{td}/m.md', 'w').write('\n'.join(md))
            subprocess.run(['pandoc', f'{td}/m.md', '-f', 'markdown+tex_math_dollars', '-o', f'{td}/m.docx'], check=True)
            xml = zipfile.ZipFile(f'{td}/m.docx').read('word/document.xml')
        root = etree.fromstring(xml)
        paras = root.findall('.//w:body/w:p', NS)
        assert len(paras) == len(inl) + len(disp), (len(paras), len(inl), len(disp))
        for s, p in zip(inl, paras[:len(inl)]):
            om = p.find('.//m:oMath', NS)
            if om is None:
                raise ValueError(f'pandoc failed on inline math: {s}')
            self.inline[s] = om
        for s, p in zip(disp, paras[len(inl):]):
            om = p.find('.//m:oMath', NS)
            if om is None:
                raise ValueError(f'pandoc failed on display math: {s}')
            self.display[s] = om


def _font_math(om):
    for r in om.iter(f'{{{M}}}r'):
        rpr = r.find('w:rPr', NS)
        if rpr is None:
            rpr = _el('w:rPr')
            mrpr = r.find('m:rPr', NS)
            if mrpr is not None:
                mrpr.addnext(rpr)
            else:
                r.insert(0, rpr)
        f = _el('w:rFonts', w_ascii='Cambria Math', w_hAnsi='Cambria Math')
        rpr.insert(0, f)
    return om


TOKEN = re.compile(r'(\$[^$]+\$|\*\*[^*]+\*\*|\*[^*]+\*)')


class Doc:
    def __init__(self, template):
        self.d = docx.Document(template)
        body = self.d.element.body
        sects = body.findall('.//w:sectPr', NS)
        # template sections: [0] single-column title block, [1] two-column body
        self.sp_single = copy.deepcopy(sects[0])
        self.sp_double = copy.deepcopy(sects[1])
        for sp in (self.sp_single, self.sp_double):
            for ref in sp.findall('w:headerReference', NS) + sp.findall('w:footerReference', NS):
                sp.remove(ref)
            t = sp.find('w:type', NS)
            if t is not None:
                sp.remove(t)
        for el in list(body):
            body.remove(el)
        self.body = body
        self.math = MathCache()
        self.items = []          # deferred content (to batch-convert math)
        self.eq_no = 0
        self.fig_no = 0
        self.tab_no = 0
        self.sec_no = 0
        self.sub_no = 0
        self.labels = {}

    # ------------------------------------------------------------------ queueing
    def _q(self, kind, **kw):
        self.items.append((kind, kw))

    def title(self, text): self._q('styled', style='Paper Title', text=text)
    def authors(self, text): self._q('styled', style='Authors', text=text)
    def affil(self, text): self._q('styled', style='Affiliations', text=text)
    def abstract(self, text): self._q('lead', lead='Abstract—', text=text)
    def keywords(self, text): self._q('lead', lead='Index Terms—', text=text)
    def end_front(self): self._q('sect', sp='single', start='nextPage')
    def section(self, text, numbered=True): self._q('section', text=text, numbered=numbered)
    def subsection(self, text): self._q('subsection', text=text)
    def para(self, text, indent=True): self._q('para', text=text, indent=indent)
    def eq(self, latex, label=None): self._q('eq', latex=latex, label=label)
    def figure(self, path, caption, wide=False, width=None, label=None): self._q('figure', path=path, caption=caption, wide=wide, width=width, label=label)
    def table(self, rows, caption, widths=None, wide=False, label=None, fs=8, header_rows=1, notes=None):
        self._q('table', rows=rows, caption=caption, widths=widths, wide=wide, label=label, fs=fs, hr=header_rows, notes=notes)
    def algorithm(self, title, lines, wide=False): self._q('algo', title=title, lines=lines, wide=wide)
    def references(self, refs): self._q('refs', refs=refs)
    def unnumbered(self, text): self._q('section', text=text, numbered=False)

    # ------------------------------------------------------------------ helpers
    def _collect_math(self):
        inl, disp = [], []

        def scan(t):
            for tok in TOKEN.findall(t or ''):
                if tok.startswith('$'):
                    inl.append(tok[1:-1])
        for k, kw in self.items:
            if k in ('para', 'lead', 'section', 'subsection'):
                scan(kw.get('text'))
            elif k == 'eq':
                disp.append(kw['latex'])
            elif k == 'figure':
                scan(kw['caption'])
            elif k == 'table':
                scan(kw['caption'])
                for r in kw['rows']:
                    for c in r:
                        scan(str(c))
                for n in (kw.get('notes') or []):
                    scan(n)
            elif k == 'algo':
                scan(kw['title'])
                for ln in kw['lines']:
                    scan(ln[1] if isinstance(ln, tuple) else ln)
        self.math.convert(inl, disp)

    def _p(self, style=None, align=None):
        p = _el('w:p')
        ppr = _el('w:pPr')
        p.append(ppr)
        if style:
            sid = self.d.styles[style].style_id
            ppr.append(_el('w:pStyle', w_val=sid))
        if align:
            ppr.append(_el('w:jc', w_val=align))
        return p

    def _run(self, text, b=False, i=False, sz=None, color=None):
        r = _el('w:r')
        rpr = _el('w:rPr')
        if b: rpr.append(_el('w:b'))
        if i: rpr.append(_el('w:i'))
        if color: rpr.append(_el('w:color', w_val=color))
        if sz: rpr.append(_el('w:sz', w_val=str(int(sz * 2))))
        if len(rpr): r.append(rpr)
        t = _el('w:t'); t.text = text
        t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        r.append(t)
        return r

    def _rich(self, p, text, sz=None, b=False, i=False):
        for tok in TOKEN.split(text):
            if not tok:
                continue
            if tok.startswith('$') and tok.endswith('$') and len(tok) > 1:
                om = copy.deepcopy(self.math.inline[tok[1:-1]])
                _font_math(om)
                if sz:
                    for r in om.iter(f'{{{M}}}r'):
                        rpr = r.find('w:rPr', NS)
                        rpr.append(_el('w:sz', w_val=str(int(sz * 2))))
                p.append(om)
            elif tok.startswith('**'):
                p.append(self._run(tok[2:-2], b=True, i=i, sz=sz))
            elif tok.startswith('*') and len(tok) > 2:
                p.append(self._run(tok[1:-1], i=True, b=b, sz=sz))
            else:
                p.append(self._run(tok, b=b, i=i, sz=sz))
        return p

    def _sect_par(self, which, start='continuous'):
        p = _el('w:p')
        ppr = _el('w:pPr')
        sp = copy.deepcopy(self.sp_single if which == 'single' else self.sp_double)
        t = _el('w:type', w_val=start)
        sp.insert(0, t)
        ppr.append(sp)
        p.append(ppr)
        return p

    def _wide_open(self):
        self.body.append(self._sect_par('double'))

    def _wide_close(self):
        self.body.append(self._sect_par('single'))

    # ------------------------------------------------------------------ render
    def render(self, out):
        self._collect_math()
        for kind, kw in self.items:
            getattr(self, '_r_' + kind)(**kw)
        # final section: two-column body
        sp = copy.deepcopy(self.sp_double)
        sp.insert(0, _el('w:type', w_val='continuous'))
        self.body.append(sp)
        self.d.save(out)
        self._fix_footer_parts(out)

    def _fix_footer_parts(self, path):
        """Blank every footer/header part in the package so no text or page field remains."""
        tmp = path + '.tmp'
        zin = zipfile.ZipFile(path)
        zout = zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED)
        for it in zin.infolist():
            data = zin.read(it.filename)
            if re.match(r'word/(footer|header)\d*\.xml', it.filename):
                root = etree.fromstring(data)
                for p in root.findall('.//w:p', NS):
                    for ch in list(p):
                        if ch.tag != f'{{{W}}}pPr':
                            p.remove(ch)
                data = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
            zout.writestr(it, data)
        zout.close(); zin.close()
        os.replace(tmp, path)

    def _r_styled(self, style, text):
        p = self._p(style)
        self._rich(p, text)
        self.body.append(p)

    def _r_lead(self, lead, text):
        p = self._p()
        p.find('w:pPr', NS).append(_el('w:ind', w_left='360', w_right='360'))
        p.find('w:pPr', NS).append(_el('w:jc', w_val='both'))
        p.append(self._run(lead, b=True, i=True))
        self._rich(p, text, b=True)
        self.body.append(p)

    def _r_sect(self, sp, start):
        self.body.append(self._sect_par(sp, start))

    def _r_section(self, text, numbered):
        if numbered:
            self.sec_no += 1
            self.sub_no = 0
            text = f'{ROMAN[self.sec_no - 1]}. {text.upper()}'
        else:
            text = text.upper()
        p = self._p('IEEE Section')
        self._rich(p, text)
        self.body.append(p)

    def _r_subsection(self, text):
        self.sub_no += 1
        p = self._p('IEEE Subsection')
        self._rich(p, f'{chr(64 + self.sub_no)}. {text}')
        self.body.append(p)

    def _r_para(self, text, indent):
        p = self._p(align='both')
        if indent:
            p.find('w:pPr', NS).append(_el('w:ind', w_firstLine='259'))
        self._rich(p, text)
        self.body.append(p)

    def _r_eq(self, latex, label):
        self.eq_no += 1
        p = self._p(align='both')
        omp = _el('m:oMathPara')
        om = _el('m:oMath')
        mm = _el('m:m')
        mpr = _el('m:mPr')
        mpr.append(_el('m:plcHide', m_val='1'))
        mcs = _el('m:mcs'); mc = _el('m:mc'); mcpr = _el('m:mcPr')
        mcpr.append(_el('m:count', m_val='4')); mcpr.append(_el('m:mcJc', m_val='center'))
        mc.append(mcpr); mcs.append(mc); mpr.append(mcs)
        ctrl = _el('m:ctrlPr'); crp = _el('w:rPr'); crp.append(_el('w:rFonts', w_ascii='Cambria Math', w_hAnsi='Cambria Math')); ctrl.append(crp)
        mpr.append(ctrl)
        mm.append(mpr)
        mr = _el('m:mr')
        e1 = _el('m:e'); e2 = _el('m:e'); e3 = _el('m:e'); e4 = _el('m:e')
        src = copy.deepcopy(self.math.display[latex])
        _font_math(src)
        for ch in list(src):
            e2.append(ch)
        nr = _el('m:r'); nrpr = _el('m:rPr'); nrpr.append(_el('m:nor')); nr.append(nrpr)
        nt = _el('m:t'); nt.text = f'({self.eq_no})'; nr.append(nt)
        e4.append(nr)
        for e in (e1, e2, e3, e4):
            mr.append(e)
        mm.append(mr)
        om.append(mm)
        omp.append(om)
        p.append(omp)
        self.body.append(p)

    def _image_par(self, path, width_in):
        p = self.d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(path, width=Inches(width_in))
        el = p._p
        el.getparent().remove(el)
        return el

    def _r_figure(self, path, caption, wide, width, label):
        self.fig_no += 1
        if wide:
            self._wide_open()
        self.body.append(self._image_par(path, width or (6.9 if wide else 3.45)))
        p = self._p('Figure Caption')
        p.find('w:pPr', NS).append(_el('w:jc', w_val='both' if wide else 'both'))
        self._rich(p, f'Fig. {self.fig_no}. ' + caption)
        self.body.append(p)
        if wide:
            self._wide_close()

    def _cell(self, tc_text, width, shade=False, fs=8, bold=False, align='center', keep=True):
        tc = _el('w:tc')
        tcpr = _el('w:tcPr')
        tcpr.append(_el('w:tcW', w_w=str(width), w_type='dxa'))
        if shade:
            tcpr.append(_el('w:shd', w_val='clear', w_color='auto', w_fill='E7E6E6'))
        tcpr.append(_el('w:vAlign', w_val='center'))
        tc.append(tcpr)
        p = _el('w:p'); ppr = _el('w:pPr'); ppr.append(_el('w:spacing', w_after='0', w_line='228', w_lineRule='auto'))
        if keep:
            ppr.insert(0, _el('w:keepNext'))
        ppr.append(_el('w:jc', w_val=align)); p.append(ppr)
        self._rich(p, str(tc_text), sz=fs, b=bold)
        tc.append(p)
        return tc

    def _r_table(self, rows, caption, widths, wide, label, fs, hr, notes):
        self.tab_no += 1
        if wide:
            self._wide_open()
        p = self._p('Table Caption')
        p.find('w:pPr', NS).append(_el('w:keepNext'))
        p.append(self._run(f'TABLE {ROMAN[self.tab_no - 1]}'))
        r = _el('w:r'); r.append(_el('w:br')); p.append(r)
        self._rich(p, caption.upper() if '$' not in caption else caption)
        self.body.append(p)
        total = 10368 if wide else 5040
        n = len(rows[0])
        if widths is None:
            widths = [1.0 / n] * n
        s = sum(widths)
        wd = [int(total * w / s) for w in widths]
        tbl = _el('w:tbl')
        tpr = _el('w:tblPr')
        tpr.append(_el('w:tblStyle', w_val='TableGrid'))
        tpr.append(_el('w:tblW', w_w=str(sum(wd)), w_type='dxa'))
        tpr.append(_el('w:jc', w_val='center'))
        tpr.append(_el('w:tblLayout', w_type='fixed'))
        cm = _el('w:tblCellMar')
        for side in ('left', 'right'):
            cm.append(_el(f'w:{side}', w_w='57', w_type='dxa'))
        tpr.append(cm)
        tbl.append(tpr)
        grid = _el('w:tblGrid')
        for w_ in wd:
            grid.append(_el('w:gridCol', w_w=str(w_)))
        tbl.append(grid)
        for i, row in enumerate(rows):
            tr = _el('w:tr')
            trpr = _el('w:trPr'); trpr.append(_el('w:jc', w_val='center'))
            if i < hr:
                trpr.append(_el('w:tblHeader'))
            trpr.append(_el('w:cantSplit'))
            tr.append(trpr)
            for j, c in enumerate(row):
                tr.append(self._cell(c, wd[j], shade=i < hr, fs=fs, bold=i < hr, align='left' if (j == 0 and i >= hr) else 'center',
                                     keep=i < len(rows) - 1))
            tbl.append(tr)
        self.body.append(tbl)
        if notes:
            for nt in notes:
                p = self._p(align='both')
                p.find('w:pPr', NS).append(_el('w:spacing', w_before='20', w_after='60'))
                self._rich(p, nt, sz=7.5)
                self.body.append(p)
        else:
            sp = self._p(); sp.find('w:pPr', NS).append(_el('w:spacing', w_after='60')); self.body.append(sp)
        if wide:
            self._wide_close()

    def _r_algo(self, title, lines, wide):
        if wide:
            self._wide_open()
        total = 10368 if wide else 5040
        tbl = _el('w:tbl')
        tpr = _el('w:tblPr')
        tpr.append(_el('w:tblW', w_w=str(total), w_type='dxa'))
        tpr.append(_el('w:jc', w_val='center'))
        b = _el('w:tblBorders')
        for side in ('top', 'bottom'):
            b.append(_el(f'w:{side}', w_val='single', w_sz='8', w_space='0', w_color='000000'))
        tpr.append(b)
        tbl.append(tpr)
        g = _el('w:tblGrid'); g.append(_el('w:gridCol', w_w=str(total))); tbl.append(g)

        def row(text, bottom=False, bold=False, ind=0, num=None):
            tr = _el('w:tr'); trpr = _el('w:trPr'); trpr.append(_el('w:cantSplit')); tr.append(trpr)
            tc = _el('w:tc'); tcpr = _el('w:tcPr'); tcpr.append(_el('w:tcW', w_w=str(total), w_type='dxa'))
            if bottom:
                bb = _el('w:tcBorders'); bb.append(_el('w:bottom', w_val='single', w_sz='4', w_space='0', w_color='000000')); tcpr.append(bb)
            tc.append(tcpr)
            p = _el('w:p'); ppr = _el('w:pPr'); ppr.append(_el('w:spacing', w_after='0', w_line='228', w_lineRule='auto'))
            if ind:
                ppr.append(_el('w:ind', w_left=str(200 * ind)))
            p.append(ppr)
            if num is not None:
                p.append(self._run(f'{num}: ', sz=8))
            self._rich(p, text, sz=8, b=bold)
            tc.append(p); tr.append(tc); tbl.append(tr)
        row(title, bottom=True, bold=True)
        k = 0
        for ln in lines:
            if isinstance(ln, tuple):
                ind, txt = ln
            else:
                ind, txt = 0, ln
            if txt.startswith('**Input') or txt.startswith('**Output') or txt.startswith('**Require') or txt.startswith('**Ensure'):
                row(txt, ind=ind)
            else:
                k += 1
                row(txt, ind=ind, num=k)
        self.body.append(tbl)
        sp = self._p(); sp.find('w:pPr', NS).append(_el('w:spacing', w_after='60')); self.body.append(sp)
        if wide:
            self._wide_close()

    def _r_refs(self, refs):
        for i, r in enumerate(refs, 1):
            p = self._p(align='both')
            ppr = p.find('w:pPr', NS)
            ppr.append(_el('w:spacing', w_after='20', w_line='220', w_lineRule='auto'))
            ppr.append(_el('w:ind', w_left='360', w_hanging='360'))
            p.append(self._run(f'[{i}]\t', sz=8))
            self._rich(p, r, sz=8)
            self.body.append(p)
