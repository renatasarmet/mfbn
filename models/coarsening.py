#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Coarsening

Copyright (C) 2020 Alan Valejo <alanvalejo@gmail.com> All rights reserved

This program comes with ABSOLUTELY NO WARRANTY. THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS
WITH YOU.

Owner or contributors are not liable for any direct, indirect, incidental, special, exemplary, or consequential
damages, (such as loss of data or profits, and others) arising in any way out of the use of this software,
even if advised of the possibility of such damage.

This program is free software and distributed in the hope that it will be useful: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version. See the GNU General Public License for more
details. You should have received a copy of the GNU General Public License along with this program. If not,
see http://www.gnu.org/licenses/.

Giving credit to the author by citing the papers.
"""

__maintainer__ = 'Alan Valejo'
__email__ = 'alanvalejo@gmail.com'
__author__ = 'Alan Valejo'
__credits__ = ['Alan Valejo']
__homepage__ = 'https://www.alanvalejo.com.br'
__license__ = 'GNU.GPL.v3'
__docformat__ = 'markdown en'
__version__ = '0.1'
__date__ = '2020-05-05'

import sys
import numpy
import multiprocessing as mp

from models.similarity import Similarity


def modified_starmap_async(function, kwargs):
    return function(**kwargs)


class Coarsening:

    def __init__(self, source_graph, **kwargs):

        prop_defaults = {
            'reduction_factor': [0.5], 'max_levels': [3], 'matching': ['rgmb'],
            'similarity': ['common_neighbors'], 'itr': [10], 'upper_bound': [0.2], 'seed_priority': ['degree'],
            'gmv': [None], 'max_hops': 2, 'layers_to_coarse': [], 'tolerance': [0.01], 'reverse': None, 'projection': 'common_neighbors',
            'pgrd': [0.50], 'deltap': [0.35], 'deltav': [0.35], 'wmin': [0.0], 'wmax': [1.0], 'threads': 1
        }

        self.__dict__.update(prop_defaults)
        self.__dict__.update(kwargs)

        self.source_graph = source_graph
        self.hierarchy_graphs = []
        self.hierarchy_levels = []

        # Validation of list values
        for prop_name, prop_value in prop_defaults.items():
            if prop_name not in ['threads', 'max_hops', 'layers_to_coarse'] and len(getattr(self, prop_name)) == 1:
                setattr(self, prop_name, [getattr(self, prop_name)[
                        0]] * self.source_graph['layers'])

        # Parameters dimension validation
        for prop_name, prop_value in prop_defaults.items():
            if prop_name not in ['threads', 'projection', 'max_hops', 'layers_to_coarse']:
                if self.source_graph['layers'] != len(getattr(self, prop_name)):
                    print('Number of layers and ' +
                          str(prop_name) + ' do not match.')
                    sys.exit(1)

        if self.threads > mp.cpu_count():
            print('Warning: Number of defined threads (' + str(self.threads) + ') '
                  'cannot be greater than the real number of cors (' + str(
                      mp.cpu_count()) + ').\n The number of '
                  'threads was setted as ' + str(mp.cpu_count()))
            self.threads = mp.cpu_count()
            sys.exit(1)

        if self.max_hops > self.source_graph['layers']:
            print('Warning: Number of defined max_hops (' + str(self.max_hops) + ') '
                  'cannot be greater than the number of layers (' + str(
                      self.source_graph['layers']) + ').\n The number of '
                  'max_hops was setted as ' + str(self.max_hops))
            self.max_hops = self.source_graph['layers']
            sys.exit(1)

        # Matching method validation
        valid_matching = ['rgmb', 'gmb', 'mlpb',
                          'hem', 'lem', 'rm', 'mnmf', 'msvm']
        for index, matching in enumerate(self.matching):
            matching = matching.lower()
            if matching not in valid_matching:
                print('Matching ' + matching + ' method is invalid.')
                sys.exit(1)
            self.matching[index] = matching

        # Seed priority validation
        valid_seed_priority = ['strength', 'degree', 'random']
        for index, seed_priority in enumerate(self.seed_priority):
            seed_priority = seed_priority.lower()
            if seed_priority not in valid_seed_priority:
                print('Seed priotiry ' + seed_priority + ' is invalid.')
                sys.exit(1)
            self.seed_priority[index] = seed_priority

        # Reverse validation
        for index, reverse in enumerate(self.reverse):
            if reverse.lower() in ('yes', 'true', 't', 'y', '1'):
                self.reverse[index] = True
            elif reverse.lower() in ('no', 'false', 'f', 'n', '0'):
                self.reverse[index] = False
            else:
                print('Boolean value expected in -rv.')
                sys.exit(1)

        # Similarity measure validation
        valid_similarity = [
            'common_neighbors', 'weighted_common_neighbors', 'hops_common_neighbors',
            'salton', 'preferential_attachment', 'jaccard', 'weighted_jaccard',
            'adamic_adar', 'resource_allocation', 'sorensen', 'hub_promoted',
            'hub_depressed', 'leicht_holme_newman', 'newman_collaboration', 'unweight'
        ]
        for index, similarity in enumerate(self.similarity):
            similarity = similarity.lower()
            if similarity not in valid_similarity:
                print('Similarity ' + similarity + ' misure is unvalid.')
                sys.exit(1)
            self.similarity[index] = similarity

        self.projection = self.projection.lower()
        if self.projection not in valid_similarity:
            print('Projection similarity ' +
                  self.projection + ' misure is unvalid.')
            sys.exit(1)

        for layer in range(self.source_graph['layers']):
            if self.matching[layer] in ['rgmb', 'gmb', 'hem', 'lem', 'rm', 'mnmf', 'msvm']:
                # if self.gmv[layer] is not None:
                #     self.gmv[layer] = None
                #     text = 'Matching method ' + self.matching[layer]
                #     text += ' (setted in layer '
                #     text += str(layer) + ') does not accept -gmv parameter.'
                #     print(text)
                if self.reduction_factor[layer] is not None and self.reduction_factor[layer] > 0.5:
                    self.reduction_factor[layer] = 0.5
                    text = 'Matching method ' + self.matching[layer]
                    text += ' (setted in layer '
                    text += str(layer) + ') does not accept -rf > 0.5.'
                    print(text)

        # Debug edges
        # How many edges in total?
        print(f"Total number of edges: {self.source_graph.ecount()}")

        for layer in range(self.source_graph['layers']):
            vertices_id = self.source_graph['vertices_by_type'][layer]

            # How many vertices with no edges?
            degree0 = [
                vertex for vertex in vertices_id if self.source_graph.degree(vertex) == 0]
            print(
                F"Layer {layer}: {len(degree0)} vertices with no edges {degree0}")

            if self.gmv[layer] is not None:
                # Since the vertices that have no edges cannot be clustered,
                # Add it to the corresponding GMV. (Respecting the limit of the number of vertices).
                self.gmv[layer] = min(
                    self.gmv[layer]+len(degree0), self.source_graph['vertices'][layer])

            min_l1 = vertices_id[0]
            max_l1 = vertices_id[-1]

            # How many edges per pair of layers?
            for l2 in range(layer+1, self.source_graph['layers']):
                vertices_id_l2 = self.source_graph['vertices_by_type'][l2]
                min_l2 = vertices_id_l2[0]
                max_l2 = vertices_id_l2[-1]
                sum_edges = sum([sum(x[min_l2:max_l2+1])
                                 for x in self.source_graph.get_adjacency()[min_l1:max_l1+1]])
                print(f"Sum edges layers {layer} and {l2} = ", sum_edges)
        print("--------------------------------------------------")

    def run(self):

        graph = self.source_graph.copy()

        # Starting neighborhood with two hops
        hop = 2
        print(f"------------------------------ hop = {hop}")
        while True:
            level = graph['level']
            contract = False
            args = []
            layers = self.layers_to_coarse if self.layers_to_coarse else range(
                graph['layers'])
            for layer in layers:
                do_matching = True
                if self.gmv[layer] is None and level[layer] >= self.max_levels[layer]:
                    print(
                        "Layer = {layer}. Max levels reached with {level[layer]} levels.")
                    do_matching = False
                elif self.gmv[layer] and graph['vertices'][layer] <= self.gmv[layer]:
                    print(
                        f"Layer = {layer}. Minimum vertices reached with {graph['vertices'][layer]} vertices.")
                    do_matching = False

                if do_matching:
                    contract = True
                    level[layer] += 1

                    kwargs = dict(
                        reduction_factor=self.reduction_factor[layer])

                    kwargs['gmv'] = self.gmv[layer]
                    if self.matching[layer] in ['mlpb', 'gmb', 'rgmb']:
                        kwargs['vertices'] = graph['vertices_by_type'][layer]
                        kwargs['reverse'] = self.reverse[layer]
                    if self.matching[layer] in ['mlpb', 'rgmb']:
                        kwargs['seed_priority'] = self.seed_priority[layer]
                    if self.matching[layer] in ['mlpb']:
                        kwargs['upper_bound'] = self.upper_bound[layer]
                        kwargs['n'] = self.source_graph['vertices'][layer]
                        kwargs['tolerance'] = self.tolerance[layer]
                        kwargs['itr'] = self.itr[layer]
                        kwargs['hop'] = hop

                    if self.matching[layer] in ['hem', 'lem', 'rm', 'mnmf', 'msvm']:
                        graph['projection'] = getattr(Similarity(
                            graph, graph['adjlist']), self.projection)
                        one_mode_graph = graph.weighted_one_mode_projection(
                            graph['vertices_by_type'][layer], similarity=self.similarity[layer])
                        matching_function = getattr(
                            one_mode_graph, self.matching[layer])
                    else:
                        graph['similarity'] = getattr(Similarity(
                            graph, graph['adjlist']), self.similarity[layer])
                        matching_function = getattr(
                            graph, self.matching[layer])

                    # Create a args for the engine multiprocessing.pool
                    args.append([(matching_function, kwargs)])

            if contract:
                # Create pools
                pool = mp.Pool(processes=self.threads)
                processes = []
                for arg in args:
                    processes.append(pool.starmap_async(
                        modified_starmap_async, arg))

                # Merge chunked solutions
                matching = numpy.arange(graph.vcount())
                for process in processes:
                    result = process.get()[0]
                    vertices = numpy.where(result > -1)[0]
                    matching[vertices] = result[vertices]

                # Close processes
                pool.close()
                pool.join()

                # Contract current graph using the matching
                coarsened_graph = graph.contract(matching)
                coarsened_graph['level'] = level

                if coarsened_graph.vcount() == graph.vcount():
                    print(
                        f"It didn't improve. Vcount = {coarsened_graph.vcount()}. matching[vertices] = {matching[vertices]}\n")
                    if hop >= self.max_hops:
                        break
                    hop += 1  # try with one more hop
                    print(f"\n\n------------------------------ hop = {hop}\n")

                self.hierarchy_graphs.append(coarsened_graph)
                self.hierarchy_levels.append(level[:])
                graph = coarsened_graph
            else:
                print("There is no available matching.")
                break
            print("\n")
