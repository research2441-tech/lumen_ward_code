"""Order-of-appearance numeric citations: [@a,b,c] -> [1-3] / [1, 4] (at most three per bracket)."""
import re
from refs import REFS

ORDER = []


def _num(k):
    if k not in REFS:
        raise KeyError(k)
    if k not in ORDER:
        ORDER.append(k)
    return ORDER.index(k) + 1


def fmt(nums):
    nums = sorted(nums)
    if len(nums) == 3 and nums[2] - nums[0] == 2:
        return f'[{nums[0]}-{nums[2]}]'
    return '[' + ', '.join(str(n) for n in nums) + ']'


def resolve(text):
    def rep(m):
        keys = [k.strip() for k in m.group(1).split(',')]
        assert len(keys) <= 3, keys
        return fmt([_num(k) for k in keys])
    return re.sub(r'\[@([^\]]+)\]', rep, text)


def bibliography():
    return [REFS[k] for k in ORDER]
