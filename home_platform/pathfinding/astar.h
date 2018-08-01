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

#ifndef ASTAR_H
#define ASTAR_H

#include <queue>
#include <vector>
#include <limits>
#include <string.h>
#include <cmath>
#include <algorithm>

class Node {

public:
	int i;
	int j;

	Node() :
		i(0), j(0), weight(0.0), cost(0.0), priority(0.0), parent(nullptr), visited(false) {
	}

	Node(int i, int j) :
			i(i), j(j), weight(0.0), cost(0.0), priority(0.0), parent(nullptr), visited(false) {
	}
	Node(int i, int j, float w) :
			i(i), j(j), weight(0.0), cost(0.0), priority(0.0), parent(nullptr), visited(false) {
	}

	bool operator==(const Node &n2) {
		return (this->i == n2.i && this->j == n2.j);
	}

	bool operator!=(const Node &n2) {
		return (this->i != n2.i || this->j != n2.j);
	}

	float weight;
	float cost;
	float priority;
	Node* parent;
	bool visited;
};

class PathFinder {

public:
	std::vector<Node> findPath(const Node& start, const Node& goal, const char* heuristic_name, bool use_diagonals);

	PathFinder(float *weights, int dim1, int dim2);

private:
	std::vector<std::vector<Node> > nodes;

	struct OrderByPriority
	{
	    bool operator() (Node* const &n1, Node* const &n2) { return n1->priority > n2->priority; }
	};
};

#endif
