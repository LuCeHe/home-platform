/******************************************************************************
 *
 * Copyright (c) 2018, Simon Brodeur
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  - Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "astar.h"

// Chebyshev distance (or diagonal distance)
float linf_norm(const Node &n1, const Node &n2) {
	return std::max(std::abs(n1.i - n2.i), std::abs(n1.j - n2.j));
}

// Manhattan distance
float l1_norm(const Node &n1, const Node &n2) {
	return std::abs(n1.i - n2.i) + std::abs(n1.j - n2.j);
}

// Euclidean distance
float l2_norm(const Node &n1, const Node &n2) {
	return std::sqrt(std::pow(n1.i - n2.i, 2) + std::pow(n1.j - n2.j, 2));
}

PathFinder::PathFinder(float* weights, int dim1, int dim2) {
	// Fill the grid of nodes
	this->nodes.resize(dim1);
	for (int i = 0; i < dim1; ++i) {
		this->nodes[i].resize(dim2);
		for (int j = 0; j < dim2; ++j) {
			this->nodes[i][j].i = i;
			this->nodes[i][j].j = j;
			this->nodes[i][j].weight = weights[i*dim2+j];
		}
	}
}

std::vector<Node> PathFinder::findPath(const Node& start_ref, const Node& goal_ref, const char* heuristic_name, bool use_diagonals) {

	const float INF = std::numeric_limits<float>::infinity();

	// Get the dimensions of the grid
	const unsigned int h = this->nodes.size();
	const unsigned int w = this->nodes[0].size();

	// Initialize costs of all nodes
	Node start = start_ref;
	Node goal = goal_ref;
	for (unsigned int i = 0; i < h; ++i) {
		for (unsigned int j = 0; j < w; ++j) {
			this->nodes[i][j].cost = INF;
			this->nodes[i][j].visited = false;
		}
	}
	this->nodes[start.i][start.j].cost = 0.0;

	// Initialize priority queue with start node
	std::priority_queue<Node*, std::vector<Node*>, OrderByPriority> opened_queue;
	opened_queue.push(&start);

	// Initialize neighbor vector
	std::vector<Node*> nbrs;
	unsigned int maxNbrs = use_diagonals ? 8 : 4;
	for (unsigned int i = 0; i < maxNbrs; ++i) {
		nbrs.push_back(nullptr);
	}

	// Get the heuristic function
    float (*heuristic)(const Node&, const Node&);
    if (!strcmp(heuristic_name, "linf-norm"))
    	heuristic = linf_norm;
    else if (!strcmp(heuristic_name, "l1-norm"))
    	heuristic = l1_norm;
    else if (!strcmp(heuristic_name, "l2-norm"))
    	heuristic = l2_norm;
    else{
    	throw "Unknown heuristic name";
    }

	// Loop until we visited all the nodes or reached the goal
	bool goal_found = false;
	while (!opened_queue.empty()) {

		// Get the head of the priority queue
		Node* cur = opened_queue.top();
		opened_queue.pop();
		cur->visited = true;

		// Stop if we reached the goal
		if (*cur == goal) {
			goal_found = true;
			break;
		}

		// Get all valid neighbors
		nbrs[0] = (cur->i < h - 1) ? &this->nodes[cur->i+1][cur->j] : nullptr;  // top
		nbrs[1] = (cur->i > 0) ? &this->nodes[cur->i-1][cur->j] : nullptr;      // down
		nbrs[2] = (cur->j > 0) ? &this->nodes[cur->i][cur->j-1] : nullptr;      // left
		nbrs[3] = (cur->j < w - 1) ? &this->nodes[cur->i][cur->j+1] : nullptr;  // right

		if (use_diagonals){
			nbrs[4] = (cur->i < h - 1 && cur->j > 0) ? &this->nodes[cur->i+1][cur->j-1] : nullptr;      // top left
			nbrs[5] = (cur->i < h - 1 && cur->j < w - 1) ? &this->nodes[cur->i+1][cur->j+1] : nullptr;  // top right
			nbrs[6] = (cur->i > 0 && cur->j > 0) ? &this->nodes[cur->i-1][cur->j-1] : nullptr;          // down left
			nbrs[7] = (cur->i > 0 && cur->j < w - 1) ? &this->nodes[cur->i-1][cur->j+1] : nullptr;      // down right
		}

		float heuristic_cost;
		for (unsigned int i = 0; i < nbrs.size(); ++i) {
			if (nbrs[i] != nullptr && !nbrs[i]->visited && nbrs[i]->weight < INF) {
				float new_cost = cur->cost + nbrs[i]->weight;
				if (new_cost < nbrs[i]->cost) {
					heuristic_cost = heuristic(*nbrs[i], goal);

					/* NOTE: The priority of a node in the queue depends on:
					         - total cost at current node
					         - cost of the move to the neighbor node
					         - heuristic at neighbor node)                 */
					nbrs[i]->priority = new_cost + heuristic_cost;
					nbrs[i]->cost = new_cost;
					nbrs[i]->parent = cur;
					opened_queue.push(nbrs[i]);
				}
			}
		}
	}

	// Reconstruct the path from start to goal
	std::vector<Node> path;
	if (goal_found) {
		Node path_node = goal;
		while (path_node != start) {
			path.push_back(this->nodes[path_node.i][path_node.j]);
			path_node = *this->nodes[path_node.i][path_node.j].parent;
		}
		// Add start node and reverse the path
		path.push_back(this->nodes[path_node.i][path_node.j]);
		std::reverse(path.begin(), path.end());
	}

	return path;
}
