from statistics import multimode  # python8


def read_file(filename, filetype, last_index_layer_0):
    with open(f'{filename}.{filetype}', 'r') as f:
        str_file = f.read()
        list_file = str_file.split()
        list_file = list_file[:last_index_layer_0]
        return list_file


def read_ncol_file(filename, filetype, last_index_layer_0):
    with open(f'{filename}.{filetype}', 'r') as f:
        str_file = f.read()
        list_edges = str_file.split('\n')
        return list_edges


def calculate_clustering_precision_and_recall(bnoc_filename, mfbn_filename, last_index_layer_0):
    print(f"CALCULATING METRICS, filename: {mfbn_filename}")

    # Reading membership files
    list_file_bnoc = read_file(filename=f'../output_bnoc/{bnoc_filename}',
                               filetype='membership',
                               last_index_layer_0=last_index_layer_0)
    list_file_mfbn = read_file(filename=f'../output_mfbn/{mfbn_filename}',
                               filetype='membership',
                               last_index_layer_0=last_index_layer_0)

    # Storing the last index in a given community
    dict_last_index_communities = {}
    current_community = '0'
    for i in range(len(list_file_bnoc)):
        if list_file_bnoc[i] != current_community:  # If I moved to another community
            dict_last_index_communities[current_community] = i-1
            current_community = list_file_bnoc[i]
        if i == len(list_file_bnoc)-1:  # If I am the last one
            dict_last_index_communities[current_community] = i

    # Separating data to each community and calculating metrics
    last_last_index = -1
    sum_n_correct = 0
    sum_precision = 0
    sum_recall = 0
    count_communities = 0
    for current_community, current_last_index in dict_last_index_communities.items():
        # Separating lists
        current_list = list_file_mfbn[last_last_index+1:current_last_index+1]
        mode_current_list = multimode(current_list)[0]

        # Getting counts
        n_total = len(current_list)  # true positive + false positive
        n_correct = current_list.count(mode_current_list)  # true positive
        n_classified_mode = list_file_mfbn.count(
            mode_current_list)  # true positive + false negative

        # Calculating metrics
        # * Metrics details
        # Precision = true positive / (true positive + false positive)
        # --> n_correct / n_total
        # Recall = true positive / (true positive + false negative)
        # --> n_correct / n_classified_mode
        # *
        sum_n_correct += n_correct
        sum_precision += n_correct/n_total
        sum_recall += n_correct/n_classified_mode
        count_communities += 1

        # Moving last index to use in the next community
        last_last_index = current_last_index

    # Average precision of each community
    avg_precison = sum_precision/count_communities
    print(f"Average precision {avg_precison*100:.2f}%")

    # Average recall of each community
    avg_recall = sum_recall/count_communities
    print(f"Average recall {avg_recall*100:.2f}% \n")


def calculate_clustering_modularity(ncol_filename, membership_filepath, membership_filename, last_index_layer_0):
    print(
        f"CALCULATING METRICS, filename: {membership_filepath}{membership_filename}")

    # Reading membership files
    list_file_mfbn = read_file(filename=f'../{membership_filepath}{membership_filename}',
                               filetype='membership',
                               last_index_layer_0=last_index_layer_0)

    list_file_ncol = read_ncol_file(filename=f'../output_bnoc/{ncol_filename}',
                                    filetype='ncol',
                                    last_index_layer_0=last_index_layer_0)

    # Creating and filling the dict where key is another type and the values are from column 0
    dict_edges = {}
    for edge in list_file_ncol:
        e = edge.split()
        if len(e) > 0 and float(e[0]) < last_index_layer_0 and float(e[1]) >= last_index_layer_0:
            e[0] = int(e[0])
            e[1] = int(e[1])
            e[2] = float(e[2])
            dict_edges[e[1]] = dict_edges.get(e[1], []) + [(e[0], e[2])]

    # Degree of vertices
    deg = {}

    # Creating and filling adjacency list
    A = [[0 for x in range(last_index_layer_0)]
         for y in range(last_index_layer_0)]
    for key in dict_edges:
        for i in range(len(dict_edges[key])):
            u = dict_edges[key][i]
            for j in range(i+1, len(dict_edges[key])):
                v = dict_edges[key][j]
                if A[u[0]][v[0]] == 0:
                    A[u[0]][v[0]] = 1  # += u[1] + v[1]
                    A[v[0]][u[0]] = 1  # += u[1] + v[1]
                    deg[u[0]] = deg.get(u[0], 0) + 1
                    deg[v[0]] = deg.get(v[0], 0) + 1

    # Counting links
    m = (sum([sum(x) for x in A])) / 2

    # Calculating the modularity
    modularity = 0
    for i in range(last_index_layer_0):
        for j in range(last_index_layer_0):
            if i != j:
                expected = (deg.get(i, 0) * deg.get(j, 0)) / (2*m)
                diff = A[i][j] - expected
                same_community = int(list_file_mfbn[i] == list_file_mfbn[j])
                modularity += diff * same_community
    modularity /= (2*m)

    print(f"Modularity {modularity:.2f} \n")


if __name__ == "__main__":

    # (bnoc_filename, mfbn_filename)
    list_tuple_files = [
        ('tripartite-1', 'tripartite-1-1', 200),
        ('tripartite-4', 'tripartite-4-2', 200),
        ('tripartite-5', 'tripartite-5-2', 200),
        ('tripartite-5', 'tripartite-5-bi1-1', 200),
        ('4partite-1', '4partite-1-2', 200),
        ('4partite-2', '4partite-2-2', 2000),
        ('5partite-1', '5partite-1-2', 200),
        ('5partite-2', '5partite-2-2', 10000),
        ('10partite-1', '10partite-1-1', 5000)
    ]

    for files in list_tuple_files:
        calculate_clustering_precision_and_recall(
            bnoc_filename=files[0],
            mfbn_filename=files[1],
            last_index_layer_0=files[2])

        print("Original:")
        calculate_clustering_modularity(
            ncol_filename=files[0],
            membership_filepath='output_bnoc/',
            membership_filename=files[0],
            last_index_layer_0=files[2])

        print("Detected:")
        calculate_clustering_modularity(
            ncol_filename=files[0],
            membership_filepath='output_mfbn/',
            membership_filename=files[1],
            last_index_layer_0=files[2])

        print("--------")
