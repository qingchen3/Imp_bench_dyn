from ST import ST_utils
from Dtree import Dtree_utils
from timeit import default_timer as timer
from Class.Res import Res
from utils.IO import printRes, update_res_query
from utils import tree_utils
from utils.graph_utils import load_parameter
import gc

global n


if __name__ == '__main__':
    sanity_check = False  # True: switch on the sanity check; False: swith off the sanity check.
    graph_types = ['star_graph', 'random_graph', 'complete_graph', 'power_law', 'usa', 'youtube', 'stackoverflow']
    ratios = [1000, 100, 20, 10, 5]  # ur = |insertion| / |deletion|, the reverse of u_r

    for ratio in ratios:
        for test_case in graph_types:
            print("evaluating query performances:", test_case, " ratio: ", ratio)
            print("\n")
            load_parameter("%s" % test_case)

            Dtree = dict()

            ST = dict()
            ST_adj_list_group_by_levels = dict()
            ST_adj_list_group_by_levels_nt = dict()

            Dtree_res = Res()
            ST_res = Res()

            # results in previous test point
            Dtree_res_pre = Res()
            ST_res_pre = Res()

            v_set = set()
            #edges_num = 0

            workloads_reader = open("workloads/%s_ratio_%d" % (test_case, ratio), 'r')
            test_points = 100
            test_size = len(workloads_reader.readlines()) // test_points
            test_query = False
            insertion_res = 0
            deletion_res = 0
            current_operation_id = 0
            workloads_reader = open("workloads/%s_ratio_%d" % (test_case, ratio), 'r')
            for line in workloads_reader.readlines():  # updates
                items = line.rstrip().split(" ")
                if items[0] == "ins":  # insertions
                    a, b = int(items[1]), int(items[2])
                    if a == b:
                        continue
                    # print("inserting %d-%d" %(a, b), current_time)
                    # Dtree
                    v_set.add(a)
                    v_set.add(b)
                    start = timer()
                    Dtree_utils.insert_edge(a, b, Dtree)
                    Dtree_res.in_time += (timer() - start)
                    Dtree_res.in_count += 1
                    insertion_res = Dtree_res.in_time / Dtree_res.in_count

                    # remove isolated nodes from v_set.
                    if Dtree[a].parent is None and Dtree[a].size == 1:
                        v_set.remove(a)

                    # Structural tree
                    start = timer()
                    ST_utils.insert(a, b, ST, ST_adj_list_group_by_levels, ST_adj_list_group_by_levels_nt)
                    ST_res.in_time += (timer() - start)
                    ST_res.in_count += 1
                    insertion_res = ST_res.in_time / ST_res.in_count
                    if sanity_check and ST_utils.query(ST[a], ST[b]) != Dtree_utils.query_simple(Dtree[a],
                                                                                                 Dtree[b]):
                        raise ValueError("Error in insertion in ST")
                    current_operation_id += 1
                elif items[0] == "del":  # deletion
                    a, b = int(items[1]), int(items[2])
                    if a == b:
                        continue
                    # Dtree
                    start = timer()
                    Dtree_utils.delete_edge(a, b, Dtree)
                    Dtree_res.de_time += (timer() - start)
                    Dtree_res.de_count += 1
                    deletion_res = Dtree_res.de_time / Dtree_res.de_count

                    start = timer()
                    ST_utils.delete(a, b, ST, ST_adj_list_group_by_levels, ST_adj_list_group_by_levels_nt)
                    ST_res.de_time += (timer() - start)
                    ST_res.de_count += 1
                    deletion_res = ST_res.de_time / ST_res.de_count
                    if sanity_check and ST_utils.query(ST[a], ST[b]) != Dtree_utils.query_simple(Dtree[a], Dtree[b]):
                        print(ST_utils.query(ST[a], ST[b]), Dtree_utils.query_simple(Dtree[a], Dtree[b]))
                        raise ValueError("Error in deletions in ST")
                    current_operation_id += 1

                if current_operation_id % test_size == 0:
                    test_queries = tree_utils.generatePairs(v_set)
                    query_Dtree = 0

                    start = timer()
                    for (x, y) in test_queries:
                        Dtree_utils.query(Dtree[x], Dtree[y])
                    query_Dtree = timer() - start

                    query_ST = 0
                    start = timer()
                    for (x, y) in test_queries:
                        ST_utils.query(ST[x], ST[y])
                    query_ST = timer() - start

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

                    update_res_query(test_case, 'query', [current_operation_id // test_size, query_Dtree], ratio, 'Dtree')
                    update_res_query(test_case, 'query', [current_operation_id // test_size, query_ST], ratio, 'ST')

                    #output_query_res(graph_label, method, "insertions", ratio, insertion_res)
                    #output_query_res(graph_label, method, "deletions", ratio, deletion_res)

            print("finishing run the experiment for dataset: %s, ratio: %d \n" % (test_case, ratio))

            del Dtree
            del ST
            del ST_adj_list_group_by_levels
            del ST_adj_list_group_by_levels_nt
            gc.collect()





