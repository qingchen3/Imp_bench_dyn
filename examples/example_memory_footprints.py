import sys

sys.path.insert(1, '../')

from _collections import defaultdict
from ST import ST_utils
from STV import STV_utils
from utils import graph_utils
from Dtree import Dtree_utils
from Dtree.DTNode import DTNode
from timeit import default_timer as timer
from Class.Res import Res
from utils.IO import output_memory_footprint_sum
from LTV import LTV_utils
from LT import LT_utils
import HK.updates as HKupdate
import HDT.updates as HDTupdate
import HKS.updates as HKSupdate
from LzT import LzT_utils
from utils.graph_utils import load_parameter


global n


def order(a, b):
    if a < b:
        return a, b
    else:
        return b, a


if __name__ == '__main__':
    sys.setrecursionlimit(50000000)
    sanity_check = True  # True: switch on the sanity check; False: swith off the sanity check.
    test_cases = ['fb']
    methods = ['Dtree', 'HKS', 'HK', 'HDT', 'ST', 'STV', 'LT', 'LTV', 'LzT', 'LCT']
    for test_case in test_cases:
        for method in methods:
            print(test_case, method)
            load_parameter(test_case)

            Dtree = dict()

            HKS_act_occ_dict = dict()
            HKS_non_tree_edges = dict()
            HKS_tree_edges_pointers = defaultdict(list)

            HK_act_occ_dict = defaultdict(defaultdict)  # HK_act_occ_dict[i] is the act_dict_i
            HK_tree_edges_pointers = [defaultdict(list) for i in range(2 * graph_utils.max_level + 1)]
            HK_non_tree_edges = dict()
            HK_tree_edges_group_by_levels = defaultdict(set)
            HK_nontree_edges_group_by_levels = defaultdict(set)

            HDT_act_occ_dict = defaultdict(defaultdict)  # HDT_act_occ_dict[i] is the act_dict_i
            HDT_tree_edges_pointers = [defaultdict(list) for i in range(2 * graph_utils.max_level + 1)]
            HDT_tree_edges_group_by_levels = defaultdict(set)
            HDT_nontree_edges_group_by_levels = defaultdict(set)

            ST = dict()
            ST_adj_list_group_by_levels = dict()
            ST_adj_list_group_by_levels_nt = dict()

            STV = dict()  # STV
            STV_adj_list_group_by_levels = dict()

            LT = dict()
            LT_t_list_group_by_levels = dict()
            LT_nt_list_group_by_levels = dict()

            LTV = dict()
            LTV_adj_list_group_by_levels = dict()

            LzT = dict()
            LzT_t_list_group_by_levels = dict()
            LzT_nt_list_group_by_levels = dict()

            HK_res = Res()
            HKS_res = Res()
            HDT_res = Res()
            Dtree_res = Res()
            ST_res = Res()
            STV_res = Res()
            LT_res = Res()
            LTV_res = Res()
            LzT_res = Res()

            # results in previous test point
            HK_res_pre = Res()
            HKS_res_pre = Res()
            HDT_res_pre = Res()
            Dtree_res_pre = Res()
            ST_res_pre = Res()
            STV_res_pre = Res()
            LT_res_pre = Res()
            LTV_res_pre = Res()
            LzT_res_pre = Res()

            delimiter = ','
            insertions_reader = open("%s" %test_case, 'r')

            if 'path_graph' and method == 'Dtree':
                edge_num = 10000000

                print(test_case, edge_num, method)
                load_parameter("%s_%d" % (test_case, edge_num))
                Dtree = dict()
                Dtree[0] = DTNode(0)
                Dtree[0].children = set()
                for i in range(1, 10000000):
                    Dtree[i] = DTNode(i)
                    Dtree[i].parent = Dtree[i - 1]
                    Dtree[i - 1].children.add(Dtree[i])
                    Dtree[i].children = set()

                space_n, space_e = Dtree_utils.cal_total_memory_use(Dtree)
                print(test_case, method, "space: %f GB, space_n: %f GB, space_e: %f GB"
                      % ((space_n + space_e) / (1024 * 1024 * 1024), space_n / (1024 * 1024 * 1024),
                         space_e / (1024 * 1024 * 1024)))
                print()
                output_memory_footprint_sum(test_case, method, (space_n + space_e) / (1024 * 1024 * 1024), None)
                continue

            for line in insertions_reader.readlines():  # insertions
                items = line.rstrip().split(delimiter)
                if items[0] == items[1]:
                    continue
                a, b = order(int(items[0]), int(items[1]))

                if method == "Dtree":
                    # evaluate Dtree
                    start = timer()
                    Dtree_utils.insert_edge(a, b, Dtree)
                    Dtree_res.in_time += (timer() - start)
                    Dtree_res.in_count += 1

                if method == "HK":
                    # evaluate HK
                    start = timer()
                    HKupdate.insert_edge(a, b, HK_act_occ_dict, HK_tree_edges_pointers,
                                         HK_non_tree_edges, HK_tree_edges_group_by_levels,
                                         HK_nontree_edges_group_by_levels)
                    HK_res.in_time += (timer() - start)
                    HK_res.in_count += 1

                # evaluate HK simplification (HKS)
                if method == "HKS":
                    start = timer()
                    HKSupdate.insert_edge(a, b, HKS_act_occ_dict, HKS_non_tree_edges, HKS_tree_edges_pointers)
                    HKS_res.in_time += (timer() - start)
                    HKS_res.in_count += 1

                # evaluate HDT
                if method == "HDT":
                    start = timer()
                    HDTupdate.insert_edge(a, b, HDT_act_occ_dict, HDT_tree_edges_pointers,
                                          HDT_tree_edges_group_by_levels, HDT_nontree_edges_group_by_levels)
                    HDT_res.in_time += (timer() - start)
                    HDT_res.in_count += 1

                # Structural tree
                if method == "ST":
                    start = timer()
                    ST_utils.insert(a, b, ST, ST_adj_list_group_by_levels, ST_adj_list_group_by_levels_nt)
                    ST_res.in_time += (timer() - start)
                    ST_res.in_count += 1

                # Structural tree variant
                if method == "STV":
                    start = timer()
                    STV_utils.insert(a, b, STV, STV_adj_list_group_by_levels)
                    STV_res.in_time += (timer() - start)
                    STV_res.in_count += 1

                # local tree (LT)
                if method == "LT":
                    start = timer()
                    LT_utils.insert(a, b, LT, LT_nt_list_group_by_levels, LT_t_list_group_by_levels)
                    LT_res.in_time += (timer() - start)
                    LT_res.in_count += 1

                # local tree variant (LTV)
                if method == "LTV":
                    start = timer()
                    LTV_utils.insert(a, b, LTV, LTV_adj_list_group_by_levels)
                    LTV_res.in_time += (timer() - start)
                    LTV_res.in_count += 1

                # lazy local tree
                if method == "LzT":
                    start = timer()
                    LzT_utils.insert(a, b, LzT, LzT_t_list_group_by_levels, LzT_nt_list_group_by_levels)
                    LzT_res.in_time += (timer() - start)
                    LzT_res.in_count += 1

            insertions_reader.close()
            # Dtree
            if method == "Dtree":
                space_n, space_e = Dtree_utils.cal_total_memory_use(Dtree)
            # HKS
            elif method == "HKS":
                space_n, space_e = HKSupdate.cal_total_memory_use(HKS_act_occ_dict, HKS_tree_edges_pointers,
                                                                      HKS_non_tree_edges)
            # HK
            elif method == "HK":
                space_n, space_e = HKupdate.cal_total_memory_use(HK_act_occ_dict, HK_tree_edges_pointers, HK_non_tree_edges,
                                              HK_tree_edges_group_by_levels, HK_nontree_edges_group_by_levels)
            elif method == "HDT":
                space_n, space_e = HDTupdate.cal_total_memory_use(HDT_act_occ_dict, HDT_tree_edges_pointers,
                                               HDT_tree_edges_group_by_levels, HDT_nontree_edges_group_by_levels)
            elif method == "ST":
                space_n, space_e = ST_utils.cal_total_memory_use(ST, ST_adj_list_group_by_levels, ST_adj_list_group_by_levels_nt)
            elif method == "STV":
                space_n, space_e = STV_utils.cal_total_memory_use(STV, STV_adj_list_group_by_levels)
            elif method == "LT":
                space_n, space_e = LT_utils.cal_total_memory_use(LT, LT_nt_list_group_by_levels, LT_t_list_group_by_levels)
            elif method == "LTV":
                space_n, space_e = LTV_utils.cal_total_memory_use(LTV, LTV_adj_list_group_by_levels)
            elif method == "LzT":
                # the same computation for Local tree (LT)
                space_n, space_e = LT_utils.cal_total_memory_use(LzT, LzT_nt_list_group_by_levels, LzT_t_list_group_by_levels)
            print(test_case, method, "space: %f GB, space_n: %f GB, space_e: %f GB"

                  %((space_n + space_e) / (1024 * 1024 * 1024), space_n / (1024 * 1024 * 1024),
                    space_e / (1024 * 1024 * 1024)))
            print()
            output_memory_footprint_sum(test_case, method, (space_n + space_e) / (1024 * 1024 * 1024), None)

