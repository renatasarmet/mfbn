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


def calculate_clustering_precision_and_recall(bnoc_filename, mfbn_filename,
                                              last_index_layer_0, remove_vertex_degree_0=False):
    """
    * Metrics details
    Precision = true positive / (true positive + false positive)
    --> n_correct / n_total
    Recall = true positive / (true positive + false negative)
    --> n_correct / n_classified_mode
    *
    """
    print(f"CALCULATING METRICS, filename: {mfbn_filename}")

    # Reading membership files
    list_file_bnoc = read_file(filename=f'outputs/output_bnoc/{bnoc_filename}/{bnoc_filename}',
                               filetype='membership',
                               last_index_layer_0=last_index_layer_0)
    list_file_mfbn = read_file(filename=f'outputs/output_mfbn/{bnoc_filename}/{mfbn_filename}',
                               filetype='membership',
                               last_index_layer_0=last_index_layer_0)

    if remove_vertex_degree_0:
        # Reading ncol file
        list_file_ncol = read_ncol_file(filename=f'outputs/output_bnoc/{bnoc_filename}/{bnoc_filename}',
                                        filetype='ncol',
                                        last_index_layer_0=last_index_layer_0)

        # Get vertices with no connections (degree=0)
        # Start the set with all vertices in column 0
        # When we find an edge from some vertex, we take it off the set
        set_vertex_with_no_edges = set(range(last_index_layer_0))
        for edge in list_file_ncol:
            e = edge.split()
            if len(e) > 0:
                set_vertex_with_no_edges.discard(int(e[0]))
                set_vertex_with_no_edges.discard(int(e[1]))

        print(f"Removing {len(set_vertex_with_no_edges)} vertices with no connections (degree=0):",
              set_vertex_with_no_edges)

        # Cleaning list_file_bnoc and list_file_mfbn: removing set_vertex_with_no_edges
        list_file_bnoc = [v for i, v in enumerate(
            list_file_bnoc) if i not in set_vertex_with_no_edges]
        list_file_mfbn = [v for i, v in enumerate(
            list_file_mfbn) if i not in set_vertex_with_no_edges]

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
        print("current_community = ", current_community)
        # Separating lists
        current_list = list_file_mfbn[last_last_index+1:current_last_index+1]
        print("current_list=", current_list)
        mode_current_list = multimode(current_list)[0]
        print("mode_current_list=", mode_current_list)

        # Getting counts
        n_total = len(current_list)  # true positive + false positive
        n_correct = current_list.count(mode_current_list)  # true positive
        n_classified_mode = list_file_mfbn.count(
            mode_current_list)  # true positive + false negative

        # Calculating metrics
        # print(f"community {current_community} n_correct=", n_correct)
        # print("n_total = ", n_total)
        sum_n_correct += n_correct
        sum_precision += n_correct/n_total
        # print("sum_precision=", sum_precision)
        # print("n_correct=", n_correct)
        sum_recall += n_correct/n_classified_mode
        count_communities += 1

        # Moving last index to use in the next community
        last_last_index = current_last_index

    # Average precision of each community
    avg_precison = sum_precision/count_communities
    print(f"Average precision {avg_precison*100:.3f}%")

    # Average recall of each community
    avg_recall = sum_recall/count_communities
    print(f"Average recall {avg_recall*100:.3f}% \n")


def calculate_clustering_modularity(ncol_folder, ncol_filename, membership_filepath, membership_filename, last_index_layer_0):
    """
    https://arxiv.org/pdf/cond-mat/0408187.pdf
    Modularity is a property of a network and a specific proposed division of that network 
    into communities. It measures when the division is a good one, in the sense that there 
    are many edges within communities and only a few between them. 
    """
    print(
        f"CALCULATING METRICS, filename: {membership_filepath}{membership_filename}")

    # Reading membership files
    list_file_mfbn = read_file(filename=f'outputs/{membership_filepath}/{ncol_folder}/{membership_filename}',
                               filetype='membership',
                               last_index_layer_0=last_index_layer_0)

    list_file_ncol = read_ncol_file(filename=f'outputs/output_bnoc/{ncol_folder}/{ncol_filename}',
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
    for key in dict_edges:  # key is from any type but column 0
        for i in range(len(dict_edges[key])):
            u = dict_edges[key][i]  # u is from column 0 (vertex, weight)
            # looking for the common neighbors of u[0]
            for j in range(i+1, len(dict_edges[key])):
                v = dict_edges[key][j]  # j is from column 0 (vertex, weight)
                if A[u[0]][v[0]] == 0:
                    A[u[0]][v[0]] = 1  # += u[1] + v[1]
                    A[v[0]][u[0]] = 1  # += u[1] + v[1]
                    deg[u[0]] = deg.get(u[0], 0) + 1
                    deg[v[0]] = deg.get(v[0], 0) + 1

    # Links are connections between vertex of column 0
    # A link exists if two vertex from column 0 have the same neighbor from another type

    # Counting links
    m = (sum([sum(x) for x in A])) / 2

    # Calculating the modularity
    modularity = 0
    for i in range(last_index_layer_0):
        for j in range(last_index_layer_0):
            if i != j:
                expected = (deg.get(i, 0) * deg.get(j, 0)) / \
                    (2*m)  # from 0 to 1
                diff = A[i][j] - expected
                same_community = int(list_file_mfbn[i] == list_file_mfbn[j])
                modularity += diff * same_community
    modularity /= (2*m)

    print(f"Modularity {modularity:.3f} \n")


if __name__ == "__main__":

    # (folder, bnoc_filename, mfbn_filename, size layer 0)
    list_tuple_files = [
        # ('tripartite-1', 'tripartite-1', 'tripartite-1-1', 20),
        # ('tripartite-2', 'tripartite-2', 'tripartite-2-2', 200),
        # ('tripartite-2', 'tripartite-2', 'tripartite-2-3', 200),
        # ('tripartite-2', 'tripartite-2-bi-1', 'tripartite-2-bi-1-1', 200),
        # ('tripartite-2', 'tripartite-2-bi-2', 'tripartite-2-bi-2-1', 200)
        # ('tripartite-3', 'tripartite-3', 'tripartite-3-2', 200)
        # ('4partite-1', '4partite-1', '4partite-1-1', 200)
        # ('4partite-2', '4partite-2', '4partite-2-3', 200)
        # ('4partite-3', '4partite-3', '4partite-3-2', 100),
        # ('4partite-3', '4partite-3', '4partite-3-5', 100),
        # ('4partite-3', '4partite-3', '4partite-3-7', 100)
        ('g_bipartite-1', 'g_bipartite-1', 'g_bipartite-1-1', 8807)
    ]

    for files in list_tuple_files:
        print("remove_vertex_degree_0=False")
        calculate_clustering_precision_and_recall(
            bnoc_filename=files[0],
            mfbn_filename=files[2],
            last_index_layer_0=files[3],
            remove_vertex_degree_0=False)

        # print("remove_vertex_degree_0=True")
        # calculate_clustering_precision_and_recall(
        #     bnoc_filename=files[0],
        #     mfbn_filename=files[2],
        #     last_index_layer_0=files[3],
        #     remove_vertex_degree_0=True)

        # print("Original:")
        # calculate_clustering_modularity(
        #     ncol_folder=files[0],
        #     ncol_filename=files[1],
        #     membership_filepath='output_bnoc/',
        #     membership_filename=files[0],
        #     last_index_layer_0=files[3])

        # print("Detected:")
        # calculate_clustering_modularity(
        #     ncol_folder=files[0],
        #     ncol_filename=files[1],
        #     membership_filepath='output_mfbn/',
        #     membership_filename=files[2],
        #     last_index_layer_0=files[3])

        print("--------")
