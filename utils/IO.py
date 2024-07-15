from pathlib import Path
from prettytable import PrettyTable
import collections
from tempfile import NamedTemporaryFile
import shutil
import csv
from pathlib import Path
import os


def setup(testcase, start_timestamp, end_timestamp):

    # setup surviving time for different datasets
    survival_time = 1296000 # by default, 14 days

    if testcase in ['dblp', 'scholar']:
        survival_time = 5  # 5 years
    elif testcase == 'osmswitzerland':
        survival_time = 129600000
    elif testcase == 'test':
        survival_time = 10

    # First, set up test_num, the number of tests of performance are conducted
    # Second, calculate test_query_frequency = (t_e - t_s) / test_num.
    # that is how frequent we test queries performance.
    # For example, test_query_frequency = 1000000, we test every 1000000 seconds.
    if testcase in ['dblp', 'scholar', 'test']:
        test_query_frequency = 1  # run performance test once per year
        test_query_num = end_timestamp - start_timestamp + survival_time  # from year 1980 to year 2021
    else:
        test_query_num = 100  # select 100 test point, calculate the frequency of testing
        test_query_frequency = (end_timestamp - start_timestamp + 2 * survival_time) // test_query_num

    test_points = []
    for i in range(1, test_query_num + 1):
        test_points.append(start_timestamp + i * test_query_frequency)

    if testcase == 'scholar':
        test_points.append(2028)

    return survival_time, test_points


def output_average_dist_by_method(data, num, testcase, method):
    """
    :param data: accumulated distances of k snapshots
    :param num: number of snapshots
    :param testcase: graph
    :param isSmallGraph: whether or not current graph is small graph
    :return: outputing distributions of average distances to files
    """
    if method not in ['Dtree', 'HK', 'CT', 'LTV']:
        raise ValueError('unknown method')

    p = "res/dist/%s" % testcase
    if not os.path.exists(p):
        os.makedirs(p, exist_ok = True)

    writer = open("%s/%s.dat" % (p, method), 'w')
    writer.write("d freq\n")

    temp = collections.OrderedDict(sorted(data.items()))

    for d, value in temp.items():
        if value // num > 0:
            writer.write("%d %d\n" % (d, value // num))
    writer.flush()
    writer.close()


def output_average_dist(data, num, testcase, isSmallGraph):
    """
    :param data: accumulated distances of k snapshots
    :param num: number of snapshots
    :param testcase: graph
    :param isSmallGraph: whether or not current graph is small graph
    :return: outputing distributions of average distances to files
    """

    Dtree_dist_data = data[0]
    temp = collections.OrderedDict(sorted(Dtree_dist_data.items()))
    writer = open("res/dist/%s/Dtree.dat" % testcase, 'w')
    writer.write("d freq\n")
    for d, value in temp.items():
        if value // num > 0:
            writer.write("%d %d\n" % (d, value // num))
    writer.flush()
    writer.close()

    nDtree_dist_data = data[1]
    temp = collections.OrderedDict(sorted(nDtree_dist_data.items()))
    writer = open("res/dist/%s/nDtree.dat" % testcase, 'w')
    writer.write("d freq\n")
    for d, value in temp.items():
        if value // num > 0:
            writer.write("%d %d\n" % (d, value // num))
    writer.flush()
    writer.close()

    HK_dist_data = data[2]
    temp = collections.OrderedDict(sorted(HK_dist_data.items()))
    writer = open("res/dist/%s/HK.dat" % testcase, 'w')
    writer.write("d freq\n")
    for d, value in temp.items():
        if value // num > 0:
            writer.write("%d %d\n" % (d, value // num))
    writer.flush()
    writer.close()

    if isSmallGraph:
        ET_dist_data = data[3]
        temp = collections.OrderedDict(sorted(ET_dist_data.items()))
        writer = open("res/dist/%s/ET.dat" % testcase, 'w')
        writer.write("d freq\n")
        for d, value in temp.items():
            if value // num > 0:
                writer.write("%d %d\n" % (d, value // num))
        writer.flush()
        writer.close()

        opt_dist_data = data[4]
        temp = collections.OrderedDict(sorted(opt_dist_data.items()))
        writer = open("res/dist/%s/opt.dat" % testcase, 'w')
        writer.write("d freq\n")
        for d, value in temp.items():
            if value // num > 0:
                writer.write("%d %d\n" % (d, value // num))
        writer.flush()
        writer.close()


# full output of maintenance
def output2file(data, insertion_writer, deletion_writer):
    Dtree_res = data[0]
    nDtree_res = data[1]
    HK_res = data[2]
    if len(data) == 3:
        insertion_writer.write("%d %f %f %d %f %f %d %f %f "
                               "%d %f %f %d %f %f %d %f %f\n"
                               % (Dtree_res.in_te_count,
                                  Dtree_res.in_te_time,
                                  Dtree_res.in_te_time / (Dtree_res.in_te_count + 0.00001),
                                  nDtree_res.in_te_count,
                                  nDtree_res.in_te_time,
                                  nDtree_res.in_te_time / (nDtree_res.in_te_count + 0.00001),
                                  HK_res.in_te_count,
                                  HK_res.in_te_time,
                                  HK_res.in_te_time / (HK_res.in_te_count + 0.00001),
                                  Dtree_res.in_nte_count,
                                  Dtree_res.in_nte_time,
                                  Dtree_res.in_nte_time / (Dtree_res.in_nte_count + 0.00001),
                                  nDtree_res.in_nte_count,
                                  nDtree_res.in_nte_time,
                                  nDtree_res.in_nte_time / (nDtree_res.in_nte_count + 0.00001),
                                  HK_res.in_nte_count,
                                  HK_res.in_nte_time,
                                  HK_res.in_nte_time / (HK_res.in_nte_count + 0.00001)))
        insertion_writer.flush()

        deletion_writer.write("%d %f %f %d %f %f %d %f %f "
                              "%d %f %f %d %f %f %d %f %f \n"
                              % (Dtree_res.de_te_count,
                                 Dtree_res.de_te_time,
                                 Dtree_res.de_te_time / (Dtree_res.de_te_count + 0.00001),
                                 nDtree_res.de_te_count,
                                 nDtree_res.de_te_time,
                                 nDtree_res.de_te_time / (nDtree_res.de_te_count + 0.00001),
                                 HK_res.de_te_count,
                                 HK_res.de_te_time,
                                 HK_res.de_te_time / (HK_res.de_te_count + 0.00001),
                                 Dtree_res.de_nte_count,
                                 Dtree_res.de_nte_time,
                                 Dtree_res.de_nte_time / (Dtree_res.de_nte_count + 0.00001),
                                 nDtree_res.de_nte_count,
                                 nDtree_res.de_nte_time,
                                 nDtree_res.de_nte_time / (nDtree_res.de_nte_count + 0.00001),
                                 HK_res.de_nte_count,
                                 HK_res.de_nte_time,
                                 HK_res.de_nte_time / (HK_res.de_nte_count + 0.00001)))
        deletion_writer.flush()
    else:
        ET_res = data[3]
        opt_res = data[4]
        insertion_writer.write("%d %f %f %d %f %f %d %f %f %d %f %f %d %f %f "
                               "%d %f %f %d %f %f %d %f %f %d %f %f %d %f %f\n"
                               % (Dtree_res.in_te_count,
                                  Dtree_res.in_te_time,
                                  Dtree_res.in_te_time / (Dtree_res.in_te_count + 0.00001),
                                  nDtree_res.in_te_count,
                                  nDtree_res.in_te_time,
                                  nDtree_res.in_te_time / (nDtree_res.in_te_count + 0.00001),
                                  ET_res.in_te_count,
                                  ET_res.in_te_time,
                                  ET_res.in_te_time / (ET_res.in_te_count + 0.00001),
                                  HK_res.in_te_count,
                                  HK_res.in_te_time,
                                  HK_res.in_te_time / (HK_res.in_te_count + 0.00001),
                                  opt_res.in_te_count,
                                  opt_res.in_te_time,
                                  opt_res.in_te_time / (opt_res.in_te_count + 0.00001),
                                  Dtree_res.in_nte_count,
                                  Dtree_res.in_nte_time,
                                  Dtree_res.in_nte_time / (Dtree_res.in_nte_count + 0.00001),
                                  nDtree_res.in_nte_count,
                                  nDtree_res.in_nte_time,
                                  nDtree_res.in_nte_time / (nDtree_res.in_nte_count + 0.00001),
                                  ET_res.in_nte_count,
                                  ET_res.in_nte_time,
                                  ET_res.in_nte_time / (ET_res.in_nte_count + 0.00001),
                                  HK_res.in_nte_count,
                                  HK_res.in_nte_time,
                                  HK_res.in_nte_time / (ET_res.in_nte_count + 0.00001),
                                  opt_res.in_nte_count,
                                  opt_res.in_nte_time,
                                  opt_res.in_nte_time / (opt_res.in_nte_count + 0.00001)))
        insertion_writer.flush()

        deletion_writer.write("%d %f %f %d %f %f %d %f %f %d %f %f %d %f %f "
                               "%d %f %f %d %f %f %d %f %f %d %f %f %d %f %f\n"
                              % (Dtree_res.de_te_count,
                                 Dtree_res.de_te_time,
                                 Dtree_res.de_te_time / (Dtree_res.de_te_count + 0.00001),
                                 nDtree_res.de_te_count,
                                 nDtree_res.de_te_time,
                                 nDtree_res.de_te_time / (nDtree_res.de_te_count + 0.00001),
                                 ET_res.de_te_count,
                                 ET_res.de_te_time,
                                 ET_res.de_te_time / (ET_res.de_te_count + 0.00001),
                                 HK_res.de_te_count,
                                 HK_res.de_te_time,
                                 HK_res.de_te_time / (HK_res.de_te_count + 0.00001),
                                 opt_res.de_te_count,
                                 opt_res.de_te_time,
                                 opt_res.de_te_time / (opt_res.de_te_count + 0.00001),
                                 Dtree_res.de_nte_count,
                                 Dtree_res.de_nte_time,
                                 Dtree_res.de_nte_time / (Dtree_res.de_nte_count + 0.00001),
                                 nDtree_res.de_nte_count,
                                 nDtree_res.de_nte_time,
                                 nDtree_res.de_nte_time / (nDtree_res.de_nte_count + 0.00001),
                                 ET_res.de_nte_count,
                                 ET_res.de_nte_time,
                                 ET_res.de_nte_time / (ET_res.de_nte_count + 0.00001),
                                 HK_res.de_nte_count,
                                 HK_res.de_nte_time,
                                 HK_res.de_nte_time / (HK_res.de_nte_count + 0.00001),
                                 opt_res.de_nte_count,
                                 opt_res.de_nte_time,
                                 opt_res.de_nte_time / (opt_res.de_nte_count + 0.00001)))
        deletion_writer.flush()


def printRes(operation, data):
    t = PrettyTable(['operation', 'Data structure', 'count', 'time'])
    for items in data:
        if len(items) < 4:
            t.add_row([operation, items[0], items[1], items[2]])
        else:
            t.add_row([operation, items[0], items[1], items[2]])
    print(t)


def update_res_vertices_edges(testcase, result_type, data):
    if result_type == 'vertices':  # update result of query performance
        filename = 'res/vertices_%s.csv'%testcase
    elif result_type == 'edges':  # update result of Sd
        filename = 'res/edges_%s.csv' % testcase
    else:
        raise ValueError('unknown result type')

    if not Path(filename).exists():
        writer = open(filename, 'w')
        writer.write('ts,num\n')
        writer.flush()
        writer.close()

    assert Path(filename).exists()

    tempfile = NamedTemporaryFile('w+t', newline = '', delete = False)
    current_time, res = data
    with open(filename, 'r', newline = '') as csvFile, tempfile:
        reader = csv.reader(csvFile, delimiter = ',', quotechar = '"')
        writer = csv.writer(tempfile, delimiter = ',', quotechar = '"')

        latest_time = -1
        for row in reader:
            if row[0] == 'ts':
                writer.writerow(row)
                continue
            elif int(row[0]) == current_time:
                writer.writerow(data)
            else:
                writer.writerow(row)
            latest_time = int(row[0])
        if latest_time < current_time:
            writer.writerow(data)

    shutil.move(tempfile.name, filename)


def update_res_query(testcase, result_type, data, method):
    # data is in the form: [current, query_time]
    if method == 'Dtree':
        index = 1
    elif method == 'HK':
        index = 2
    elif method == 'ST':
        index = 3
    elif method == 'WN':
        index = 4
    elif method == 'TP':
        index = 5
    else:
        raise ValueError('unknown method')

    if result_type == 'query':  # update result of query performance
        filename = 'res/query_%s.csv' % testcase
    else:
        raise ValueError('unknown result type')

    if not Path(filename).exists():
        writer = open(filename, 'w')
        writer.write('ts,Dtree,HK,ST,WN,TP\n')
        writer.flush()
        writer.close()

    assert Path(filename).exists()

    tempfile = NamedTemporaryFile('w+t', newline = '', delete = False)
    current_time, res = data
    with open(filename, 'r', newline = '') as csvFile, tempfile:
        reader = csv.reader(csvFile, delimiter = ',', quotechar = '"')
        writer = csv.writer(tempfile, delimiter = ',', quotechar = '"')

        latest_time = -1
        for row in reader:
            if row[0] == 'ts':
                writer.writerow(row)
                continue
            elif int(row[0]) == current_time:
                items = []
                for i in range(len(row)):
                    items.append(row[i])
                items[index] = res
                writer.writerow(items)
            else:
                writer.writerow(row)
            latest_time = int(row[0])
        if latest_time < current_time:
            items = [current_time, 0, 0, 0, 0, 0]
            items[index] = res
            writer.writerow(items)

    shutil.move(tempfile.name, filename)


def update_average_distance(testcase, data, method):
    # data is in the form: [current, query_time]
    if method == 'Dtree':
        index = 1
    elif method == 'HK':
        index = 2
    elif method == 'CT':
        index = 3
    elif method == 'LTV':
        index = 4
    else:
        raise ValueError('unknown method')

    if not os.path.exists('./res'):
        os.makedirs('./res', exist_ok = True)

    if testcase in ['fb', 'wiki', 'dnc', 'messages', 'call']:
        is_small_graph = True
    else:
        is_small_graph = False

    filename = './res/height_%s.csv'%testcase
    if not Path(filename).exists():
        writer = open(filename, 'w')
        if is_small_graph:
            writer.write('ts,Dtree,nDtree,HK,ET,opt\n')
        else:
            writer.write('ts,Dtree,nDtree,HK\n')
        writer.flush()
        writer.close()

    assert Path(filename).exists()

    tempfile = NamedTemporaryFile('w+t', newline = '', delete = False)
    current_time, res = data
    with open(filename, 'r', newline = '') as csvFile, tempfile:
        reader = csv.reader(csvFile, delimiter = ',', quotechar = '"')
        writer = csv.writer(tempfile, delimiter = ',', quotechar = '"')

        latest_time = -1
        for row in reader:
            if row[0] == 'ts':
                writer.writerow(row)
                continue
            elif int(row[0]) == current_time:
                items = []
                for i in range(len(row)):
                    items.append(row[i])
                items[index] = res
                writer.writerow(items)
            else:
                writer.writerow(row)
            latest_time = int(row[0])
        if latest_time < current_time:
            if is_small_graph:
                items = [current_time, 0, 0, 0, 0, 0]
            else:
                items = [current_time, 0, 0, 0]
            items[index] = res
            writer.writerow(items)

    shutil.move(tempfile.name, filename)


def output_memory_footprint(testcase, method, data, u_r):  # output space_n and space_e
    # data is in the form: [method, memory_footprint for space_n, memory footprint for space_e]
    if method not in ['Dtree', 'HKS', 'HK', 'HDT', 'ST', 'STV', 'LT', 'LTV', 'LzT']:
        raise ValueError('unknown method')

    if not os.path.exists('./res/memory'):
        os.makedirs('./res/memory', exist_ok = True)
    if u_r is None:
        filename = './res/memory/%s.csv' % testcase
    else:
        filename = './res/memory/%s_%d.csv' % (testcase, u_r)

    if not Path(filename).exists():
        writer = open(filename, 'w')
        writer.write('method,sn,se\n')  # sn: space_n; se:spece_e
        writer.flush()
        writer.close()
    assert Path(filename).exists()

    tempfile = NamedTemporaryFile('w+t', newline='', delete=False)
    space_n, space_e = data
    with open(filename, 'r', newline='') as csvFile, tempfile:
        reader = csv.reader(csvFile, delimiter=',', quotechar='"')
        writer = csv.writer(tempfile, delimiter=',', quotechar='"')
        method_evalauted = False
        for row in reader:
            if row[0] == method:
                items = [method, space_n, space_e]
                writer.writerow(items)
                method_evalauted = True
            else:
                writer.writerow(row)

        if not method_evalauted:
            items = [method, space_n, space_e]
            writer.writerow(items)
    shutil.move(tempfile.name, filename)


def output_memory_footprint_sum(testcase, method, data, u_r):  # output space = space_n + space_e
    # data is in the form: [method, memory_footprint for space_n, memory footprint for space_e]
    if method not in ['Dtree', 'HKS', 'HK', 'HDT', 'ST', 'STV', 'LT', 'LTV', 'LzT']:
        raise ValueError('unknown method')

    if not os.path.exists('./res/memory'):
        os.makedirs('./res/memory', exist_ok = True)
    if u_r is None:
        filename = './res/memory/%s_sum.csv' % testcase
    else:
        filename = './res/memory/%s_sum_%d.csv' % (testcase, u_r)

    if not Path(filename).exists():
        writer = open(filename, 'w')
        writer.write('method,s\n')  # s: space_n + spece_e
        writer.flush()
        writer.close()
    assert Path(filename).exists()

    tempfile = NamedTemporaryFile('w+t', newline='', delete=False)
    space = data
    with open(filename, 'r', newline='') as csvFile, tempfile:
        reader = csv.reader(csvFile, delimiter=',', quotechar='"')
        writer = csv.writer(tempfile, delimiter=',', quotechar='"')
        method_evalauted = False
        for row in reader:
            if row[0] == method:
                items = [method, space]
                writer.writerow(items)
                method_evalauted = True
            else:
                writer.writerow(row)

        if not method_evalauted:
            items = [method, space]
            writer.writerow(items)
    shutil.move(tempfile.name, filename)


def update_res_query(testcase, result_type, data, ratio, method):
    # data is in the form: [current, query_time]
    if method == 'Dtree':
        index = 1
    elif method == 'ST':
        index = 2
    else:
        raise ValueError('method no concerned')

    if result_type == 'query':  # update result of query performance
        filename = 'res/query/query_%s_%d.csv'%(testcase, ratio)
    else:
        raise ValueError('unknown result type')

    if not os.path.exists('./res/%s' %result_type):
        os.makedirs('./res/%s' %result_type, exist_ok = True)

    if not Path(filename).exists():
        writer = open(filename, 'w')
        writer.write('no,Dtree,ST\n')
        writer.flush()
        writer.close()

    assert Path(filename).exists()

    tempfile = NamedTemporaryFile('w+t', newline = '', delete = False)
    current_idx, res = data
    with open(filename, 'r', newline = '') as csvFile, tempfile:
        reader = csv.reader(csvFile, delimiter = ',', quotechar = '"')
        writer = csv.writer(tempfile, delimiter = ',', quotechar = '"')

        latest_idx = -1
        for row in reader:
            if row[0] == 'no':
                writer.writerow(row)
                continue
            elif int(row[0]) == current_idx:
                items = []
                for i in range(len(row)):
                    items.append(row[i])
                items[index] = res
                writer.writerow(items)
            else:
                writer.writerow(row)
            latest_idx = int(row[0])
        if latest_idx < current_idx:
            items = [current_idx, 0, 0]
            items[index] = res
            writer.writerow(items)

    shutil.move(tempfile.name, filename)
    return


def output_runtine(graph_label, method, operation, ratio, data):
    if method not in ['Dtree', 'HKS', 'HK', 'HDT', 'ST', 'STV', 'LT', 'LTV', 'LzT']:
        raise ValueError('unknown method')
    if operation != 'insertions' and operation != 'deletions':
        raise ValueError('invalid input for operations')

    method_index = 0
    if method == 'Dtree':
        method_index = 1
    elif method == 'HKS':
        method_index = 2
    elif method == 'HK':
        method_index = 3
    elif method == 'HDT':
        method_index = 4
    elif method == 'ST':
        method_index = 5
    elif method == 'STV':
        method_index = 6
    elif method == 'LT':
        method_index = 7
    elif method == 'LTV':
        method_index = 8
    elif method == 'LzT':
        method_index = 9
    else:
        raise ValueError('unknown method')

    if not os.path.exists('./res/%s' %operation):
        os.makedirs('./res/%s' %operation, exist_ok = True)

    filename = './res/%s/%s_%d.csv' % (operation, operation, ratio)

    if not Path(filename).exists():
        writer = open(filename, 'w')
        writer.write('Graph,Dtree,HKS,HK,HDT,ST,STV,LT,LTV,LzT\n')
        writer.write('SG,0,0,0,0,0,0,0,0,0\n')
        writer.write('PG,0,0,0,0,0,0,0,0,0\n')
        writer.write('CG,0,0,0,0,0,0,0,0,0\n')
        writer.write('RG,0,0,0,0,0,0,0,0,0\n')
        writer.write('PL,0,0,0,0,0,0,0,0,0\n')
        writer.write('YT,0,0,0,0,0,0,0,0,0\n')
        writer.write('ST,0,0,0,0,0,0,0,0,0\n')
        writer.write('USA,0,0,0,0,0,0,0,0,0\n')
        writer.write('TK,0,0,0,0,0,0,0,0,0\n')
        writer.flush()
        writer.close()
    assert Path(filename).exists()

    tempfile = NamedTemporaryFile('w+t', newline='', delete=False)

    with open(filename, 'r', newline='') as csvFile, tempfile:
        reader = csv.reader(csvFile, delimiter=',', quotechar='"')
        writer = csv.writer(tempfile, delimiter=',', quotechar='"')
        for row in reader:
            if row[0] == graph_label:
                row[method_index] = data
                writer.writerow(row)
            else:
                writer.writerow(row)
    shutil.move(tempfile.name, filename)


def update_average_runtime(testcase, type, data, method):
    # data is in the form: [current, query_time]
    if method == 'Dtree':
        index = 1
    elif method == 'HK':
        index = 2
    elif method == 'ST':
        index = 3
    elif method == 'WN':
        index = 4
    elif method == 'TP':
        index = 5
    else:
        raise ValueError('unknown method')

    if not os.path.exists('./res/updates'):
        os.makedirs('./res/updates', exist_ok = True)

    if type == "insertion":
        filename = './res/updates/%s_insertion.csv' % testcase
    elif type == "deletion":
        filename = './res/updates/%s_deletion.csv' % testcase
    else:
        raise ValueError('unknown parameters')

    if not Path(filename).exists():
        writer = open(filename, 'w')
        writer.write('ts,Dtree,HK,ST,WN,TP\n')
        writer.flush()
        writer.close()

    assert Path(filename).exists()

    tempfile = NamedTemporaryFile('w+t', newline = '', delete = False)
    current_time, res = data
    with open(filename, 'r', newline = '') as csvFile, tempfile:
        reader = csv.reader(csvFile, delimiter = ',', quotechar = '"')
        writer = csv.writer(tempfile, delimiter = ',', quotechar = '"')

        latest_time = -1
        for row in reader:
            if row[0] == 'ts':
                writer.writerow(row)
                continue
            elif int(row[0]) == current_time:
                items = []
                for i in range(len(row)):
                    items.append(row[i])
                items[index] = res
                writer.writerow(items)
            else:
                writer.writerow(row)
            latest_time = int(row[0])
        if latest_time < current_time:
            items = [current_time, 0, 0, 0, 0, 0]
            items[index] = res
            writer.writerow(items)

    shutil.move(tempfile.name, filename)


def update_maintanence(testcase, data, method):
    files = ['res/insertion.csv',  'res/deletion.csv']
    if testcase not in ['wiki', 'messages', 'dnc', 'call', 'fb', 'tech', 'enron', 'stackoverflow', 'youtube',
                        'osmswitzerland', 'test',] or 'power_law' in testcase:
        raise ValueError(" testcase not in ['wiki', 'messages', 'dnc', 'call', 'fb', "
                         "'tech', 'enron', 'stackoverflow', 'youtube', 'dblp', 'osmswitzerland', 'test']")

    if method not in ['Dtree', 'HK', 'ST', 'STV', 'WN', 'TP', 'LzT']:
        raise ValueError("method not in ['Dtree', 'HK', 'ST', 'STV', 'WN', 'TP', 'LzT']")

    labels = {'wiki': 'WI', 'messages': 'MS', 'dnc': 'DNC', 'call': 'CA', 'fb': 'FB',
            'tech': 'Tech', 'enron': 'EN', 'stackoverflow': 'SO', 'youtube': 'YT', 'osmswitzerland': 'OSM', 'test':'Test'}
    testcase_ = labels[testcase]

    method_index = {'Dtree': 1, 'HK': 2, 'ST': 3, 'WN': 4, 'TP': 5}
    idx = method_index[method]

    for filename in files:
        if not Path(filename).exists():
            writer = open(filename, 'w')
            writer.write('Graph,DTree,HK,ST,WN,TP\n')
            writer.write("DNC,0,0,0,0,0\nCA,0,0,0,0,0\n"
                         "MS,0,0,0,0,0\nWI,0,0,0,0,0\n"
                         "FB,0,0,0,0,0\nTech,0,0,0,0,0\n"
                         "EN,0,0,0,0,0\nSO,0,0,0,0,0\n"
                         "YT,0,0,0,0,0\nOSM,0,0,0,0,0")
            writer.flush()
            writer.close()

        tempfile = NamedTemporaryFile('w+t', newline = '', delete = False)
        with open(filename, 'r', newline = '') as csvFile, tempfile:
            reader = csv.reader(csvFile, delimiter = ',', quotechar = '"')
            writer = csv.writer(tempfile, delimiter = ',', quotechar = '"')
            for row in reader:
                case = row[0]
                if case != testcase_:
                    writer.writerow(row)
                else:
                    items = []
                    for k in range(len(row)):
                        items.append(row[k])
                    if "insertion" in filename:
                        items[idx] = data.in_time / (data.in_count + 0.00001)
                    elif "deletion" in filename:
                        items[idx] = (data.de_time / (data.de_count + 0.00001))
                    else:
                        raise ValueError("Wrong operations")
                    writer.writerow(items)
        shutil.move(tempfile.name, filename)


def copyRes(current_res, pre_res):
    pre_res.in_count = current_res.in_count
    pre_res.in_time = current_res.in_time

    pre_res.de_count = current_res.de_count
    pre_res.de_time = current_res.de_time

