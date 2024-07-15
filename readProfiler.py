import sys, os, math, cProfile, pstats, io
from pstats import SortKey


if __name__ == '__main__':
    sys.setrecursionlimit(50000000)
    folder = 'dataset/'
    testcase = sys.argv[1]

    v_set = set()
    edges_num = 0

    # pr = cProfile.Profile("./"+testcase + "_profiling")
    # TP_in_exclude = 0
    # s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats("./"+testcase + "_deletion_profiling").strip_dirs().sort_stats(sortby)
    # q3
    #
    #   print(ps.get_stats_profile().func_profiles['__init__'], ps.get_stats_profile().func_profiles['__init__'].cumtime)
    ps.print_stats()
    # print(s.getvalue())
