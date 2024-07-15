import sys
from _collections import defaultdict
from ST import ST_utils
from STV import STV_utils
from utils import graph_utils
from Dtree import Dtree_utils
from timeit import default_timer as timer
from Class.Res import Res
from utils.IO import output_runtine
import HK.updates as HKupdate
import HDT.updates as HDTupdate
import HKS.updates as HKSupdate
from LTV import LTV_utils
from LT import LT_utils
from LzT import LzT_utils
from utils import tree_utils
from utils.graph_utils import load_parameter
import gc

global n


if __name__ == '__main__':
    sys.setrecursionlimit(50000000)
    method = sys.argv[1]  # the data structure to be evaluated
    sanity_check = False  # True: switch on the sanity check; False: swith off the sanity check.
    graph_types = ['star_graph', 'path_graph', 'random_graph', 'complete_graph', 'power_law', 'usa', 'youtube', 'stackoverflow']
    ratios = [1000, 100, 20, 10, 5]  # ur = |insertion| / |deletion|, the reverse of u_r

    for ratio in ratios:
        for test_case in graph_types:
            load_parameter("%s" % test_case)

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

            workloads_reader = open("workloads/%s_ratio_%d" % (test_case, ratio), 'r')

            test_query = False
            insertion_res = 0
            deletion_res = 0
            for line in workloads_reader.readlines():  # updates
                items = line.rstrip().split(" ")
                if items[0] == "ins":  # insertions
                    a, b = int(items[1]), int(items[2])
                    if a == b:
                        continue
                    # print("inserting %d-%d" %(a, b), current_time)
                    # Dtree
                    if method == "Dtree":
                        start = timer()
                        Dtree_utils.insert_edge(a, b, Dtree)
                        Dtree_res.in_time += (timer() - start)
                        Dtree_res.in_count += 1
                        insertion_res = Dtree_res.in_time / Dtree_res.in_count
                    elif method == "HK":
                        # evaluate HK
                        start = timer()
                        HKupdate.insert_edge(a, b, HK_act_occ_dict, HK_tree_edges_pointers,
                                             HK_non_tree_edges, HK_tree_edges_group_by_levels,
                                             HK_nontree_edges_group_by_levels)
                        HK_res.in_time += (timer() - start)
                        HK_res.in_count += 1
                        insertion_res = HK_res.in_time / HK_res.in_count
                        if sanity_check and Dtree_utils.query_simple(Dtree[a], Dtree[b]) != HKupdate.query(a, b,
                                                                                                           HK_act_occ_dict):
                            raise ValueError("Error in insertions on HK")
                    elif method == "HKS":
                        start = timer()
                        HKSupdate.insert_edge(a, b, HKS_act_occ_dict, HKS_non_tree_edges, HKS_tree_edges_pointers)
                        HKS_res.in_time += (timer() - start)
                        HKS_res.in_count += 1
                        insertion_res = HKS_res.in_time / HKS_res.in_count
                        if sanity_check and Dtree_utils.query_simple(Dtree[a], Dtree[b]) != tree_utils.query(a, b,
                                                                                                             HKS_act_occ_dict):
                            raise ValueError("Error in insertions on HK variant")
                    elif method == "HDT":
                        start = timer()
                        HDTupdate.insert_edge(a, b, HDT_act_occ_dict, HDT_tree_edges_pointers,
                                              HDT_tree_edges_group_by_levels, HDT_nontree_edges_group_by_levels)
                        HDT_res.in_time += (timer() - start)
                        HDT_res.in_count += 1
                        insertion_res = HDT_res.in_time / HDT_res.in_count
                        if sanity_check and Dtree_utils.query_simple(Dtree[a], Dtree[b]) != HDTupdate.query(a, b,
                                                                                                            HDT_act_occ_dict):
                            raise ValueError("Error in insertions on HDT")
                    elif method == "ST":
                        # Structural tree
                        start = timer()
                        ST_utils.insert(a, b, ST, ST_adj_list_group_by_levels, ST_adj_list_group_by_levels_nt)
                        ST_res.in_time += (timer() - start)
                        ST_res.in_count += 1
                        insertion_res = ST_res.in_time / ST_res.in_count
                        if sanity_check and ST_utils.query(ST[a], ST[b]) != Dtree_utils.query_simple(Dtree[a],
                                                                                                     Dtree[b]):
                            raise ValueError("Error in insertion in ST")
                    elif method == "STV":
                        # Structural tree variant
                        start = timer()
                        STV_utils.insert(a, b, STV, STV_adj_list_group_by_levels)
                        STV_res.in_time += (timer() - start)
                        STV_res.in_count += 1
                        insertion_res = STV_res.in_time / STV_res.in_count
                        if sanity_check and STV_utils.query(ST[a], ST[b]) != Dtree_utils.query_simple(Dtree[a],
                                                                                                             Dtree[b]):
                            raise ValueError("Error in insertion in ST variant")
                    elif method == "LT":
                        start = timer()
                        LT_utils.insert(a, b, LT, LT_nt_list_group_by_levels, LT_t_list_group_by_levels)
                        LT_res.in_time += (timer() - start)
                        LT_res.in_count += 1
                        insertion_res = LT_res.in_time / LT_res.in_count
                        if sanity_check and LTV_utils.query(LT, a, b) != Dtree_utils.query_simple(Dtree[a], Dtree[b]):
                            raise ValueError("Error in insertion in LT")
                    elif method == "LTV":
                        start = timer()
                        LTV_utils.insert(a, b, LTV, LTV_adj_list_group_by_levels)
                        LTV_res.in_time += (timer() - start)
                        LTV_res.in_count += 1
                        insertion_res = LTV_res.in_time / LTV_res.in_count
                        if sanity_check and LTV_utils.query(LTV, a, b) != Dtree_utils.query_simple(Dtree[a], Dtree[b]):
                            raise ValueError("Error in insertion in WN")
                    elif method == "LzT":
                        start = timer()
                        LzT_utils.insert(a, b, LzT, LzT_t_list_group_by_levels, LzT_nt_list_group_by_levels)
                        LzT_res.in_time += (timer() - start)
                        LzT_res.in_count += 1
                        insertion_res = LzT_res.in_time / LzT_res.in_count
                        if sanity_check and LzT_utils.query(LzT, a, b) != Dtree_utils.query_simple(Dtree[a], Dtree[b]):
                            raise ValueError("Error in insertions in LzT")
                    else:
                        raise ValueError("Wrong method ...")
                elif items[0] == "del":  # deletion
                    a, b = int(items[1]), int(items[2])
                    if a == b:
                        continue
                    if method == "Dtree":
                        # Dtree
                        start = timer()
                        Dtree_utils.delete_edge(a, b, Dtree)
                        Dtree_res.de_time += (timer() - start)
                        Dtree_res.de_count += 1
                        deletion_res = Dtree_res.de_time / Dtree_res.de_count
                    elif method == "HKS":
                        start = timer()
                        HKSupdate.delete_edge(a, b, HKS_act_occ_dict, HKS_non_tree_edges, HKS_tree_edges_pointers)
                        HKS_res.de_time += (timer() - start)
                        HKS_res.de_count += 1
                        deletion_res = HKS_res.de_time / HKS_res.de_count
                        if sanity_check and Dtree_utils.query_simple(Dtree[a], Dtree[b]) != \
                                tree_utils.query(a, b, HKS_act_occ_dict):
                            print(tree_utils.query(a, b, HKS_act_occ_dict),
                                  Dtree_utils.query_simple(Dtree[a], Dtree[b]))
                            raise ValueError("Error in deletions on HK variant")
                    elif method == "HK":
                        # HK
                        start = timer()
                        HKupdate.delete_edge(a, b, HK_act_occ_dict, HK_tree_edges_pointers,
                                             HK_non_tree_edges, HK_tree_edges_group_by_levels,
                                             HK_nontree_edges_group_by_levels)
                        HK_res.de_time += (timer() - start)
                        HK_res.de_count += 1
                        deletion_res = HK_res.de_time / HK_res.de_count
                        if sanity_check and Dtree_utils.query_simple(Dtree[a], Dtree[b]) != \
                                HKupdate.query(a, b, HK_act_occ_dict):
                            raise ValueError("Error in deletions on HK")
                    elif method == "HDT":
                        # HDT
                        start = timer()
                        HDTupdate.delete_edge(a, b, HDT_act_occ_dict, HDT_tree_edges_pointers,
                                              HDT_tree_edges_group_by_levels, HDT_nontree_edges_group_by_levels)
                        HDT_res.de_time += (timer() - start)
                        HDT_res.de_count += 1
                        deletion_res = HDT_res.de_time / HDT_res.de_count
                        if sanity_check and Dtree_utils.query_simple(Dtree[a], Dtree[b]) != HDTupdate.query(a, b,
                                                                                                            HDT_act_occ_dict):
                            print(Dtree_utils.query_simple(Dtree[a], Dtree[b]), HDTupdate.query(a, b, HDT_act_occ_dict))
                            raise ValueError("Error in deletions on HDT")
                    elif method == "ST":
                        start = timer()
                        ST_utils.delete(a, b, ST, ST_adj_list_group_by_levels, ST_adj_list_group_by_levels_nt)
                        ST_res.de_time += (timer() - start)
                        ST_res.de_count += 1
                        deletion_res = ST_res.de_time / ST_res.de_count
                        if sanity_check and ST_utils.query(ST[a], ST[b]) != Dtree_utils.query_simple(Dtree[a], Dtree[b]):
                            print(ST_utils.query(ST[a], ST[b]), Dtree_utils.query_simple(Dtree[a], Dtree[b]))
                            raise ValueError("Error in deletions in ST")
                    elif method == "STV":
                        # Structural Tree variant
                        start = timer()
                        STV_utils.delete(a, b, STV, STV_adj_list_group_by_levels)
                        STV_res.de_time += (timer() - start)
                        STV_res.de_count += 1
                        deletion_res = STV_res.de_time / STV_res.de_count
                        if sanity_check and STV_utils.query(ST[a], ST[b]) != Dtree_utils.query_simple(Dtree[a],
                                                                                                             Dtree[b]):
                            print(STV_utils.query(ST[a], ST[b]), Dtree_utils.query_simple(Dtree[a], Dtree[b]))
                            raise ValueError("Error in deletions in ST varinat")
                    elif method == "LT":
                        # LT
                        start = timer()
                        LT_utils.delete(a, b, LT, LT_nt_list_group_by_levels, LT_t_list_group_by_levels)
                        LT_res.de_time += (timer() - start)
                        LT_res.de_count += 1
                        deletion_res = LT_res.de_time / LT_res.de_count
                        if sanity_check and LTV_utils.query(LT, a, b) != Dtree_utils.query_simple(Dtree[a], Dtree[b]):
                            raise ValueError("Error in deletions in LT")
                    elif method == "LTV":
                        start = timer()
                        LTV_utils.delete(a, b, LTV, LTV_adj_list_group_by_levels)
                        LTV_res.de_time += (timer() - start)
                        LTV_res.de_count += 1
                        deletion_res = LTV_res.de_time / LTV_res.de_count
                        if sanity_check and LTV_utils.query(LTV, a, b) != Dtree_utils.query_simple(Dtree[a], Dtree[b]):
                            raise ValueError("Error in deletions in WN")
                    elif method == "LzT":
                        start = timer()
                        LzT_utils.delete(a, b, LzT, LzT_t_list_group_by_levels, LzT_nt_list_group_by_levels)
                        LzT_res.de_time += (timer() - start)
                        LzT_res.de_count += 1
                        deletion_res = LzT_res.de_time / LzT_res.de_count
                        if sanity_check and LzT_utils.query(LzT, a, b) != Dtree_utils.query_simple(Dtree[a], Dtree[b]):
                            raise ValueError("Error in deletions in LzT")
                    else:
                        raise ValueError("Wrong method")

            # prepare output label
            if test_case == "star_graph":
                graph_label = "SG"
            elif test_case == "complete_graph":
                graph_label = "CG"
            elif test_case == "path_graph":
                graph_label = "PG"
            elif test_case == "random_graph":
                graph_label = "RG"
            elif test_case == "power_law":
                graph_label = "PL"
            elif test_case == "trackers":
                graph_label = "TK"
            elif test_case == "stackoverflow":
                graph_label = "ST"
            elif test_case == "youtube":
                graph_label = "YT"
            elif test_case == "usa":
                graph_label = "USA"
            elif test_case == "SC":
                graph_label = "SC"
            else:
                raise ValueError("Wrong dataset")

            output_runtine(graph_label, method, "insertions", ratio, insertion_res)
            output_runtine(graph_label, method, "deletions", ratio, deletion_res)
            print("insertion res: %f; deletion res: %f" % (insertion_res, deletion_res))
            print("finishing run the experiment for dataset: %s, ratio: %d, method: %s \n" % (test_case, ratio, method))

            del Dtree

            del HKS_act_occ_dict
            del HKS_non_tree_edges
            del HKS_tree_edges_pointers

            del HK_act_occ_dict
            del HK_tree_edges_pointers
            del HK_non_tree_edges
            del HK_tree_edges_group_by_levels
            del HK_nontree_edges_group_by_levels

            del HDT_act_occ_dict
            del HDT_tree_edges_pointers
            del HDT_tree_edges_group_by_levels
            del HDT_nontree_edges_group_by_levels

            del ST
            del ST_adj_list_group_by_levels
            del ST_adj_list_group_by_levels_nt

            del STV
            del STV_adj_list_group_by_levels

            del LT
            del LT_t_list_group_by_levels
            del LT_nt_list_group_by_levels

            del LTV
            del LTV_adj_list_group_by_levels

            del LzT
            del LzT_t_list_group_by_levels
            del LzT_nt_list_group_by_levels

            gc.collect()





