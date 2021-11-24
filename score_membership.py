from statistics import multimode  # python8


def read_file(filename, filetype, last_index_layer_0):
    with open(f'{filename}.{filetype}', 'r') as f:
        str_file = f.read()
        list_file = str_file.split()
        list_file = list_file[:last_index_layer_0]
        return list_file


def calculate_clustering_precision_and_recall(bnoc_filename, mfbn_filename):
    print(f"CALCULATING METRICS, filename: {mfbn_filename}")

    # Reading membership files
    list_file_bnoc = read_file(filename=f'../output_bnoc/{bnoc_filename}',
                               filetype='membership',
                               last_index_layer_0=200)
    list_file_mfbn = read_file(filename=f'../output_mfbn/{mfbn_filename}',
                               filetype='membership',
                               last_index_layer_0=200)

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


if __name__ == "__main__":

    # (bnoc_filename, mfbn_filename)
    list_tuple_files = [
        ('tripartite-1', 'tripartite-1-1'),
        ('tripartite-4', 'tripartite-4-2'),
        ('tripartite-5', 'tripartite-5-2'),
        ('tripartite-5', 'tripartite-5-bi1-1'),
        ('4partite-1', '4partite-1-2'),
        ('4partite-2', '4partite-2-2'),
        ('5partite-1', '5partite-1-2'),
        ('5partite-2', '5partite-2-2'),
        ('10partite-1', '10partite-1-1')
    ]

    for files in list_tuple_files:
        calculate_clustering_precision_and_recall(
            bnoc_filename=files[0],
            mfbn_filename=files[1])
