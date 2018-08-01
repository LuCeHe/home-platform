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

#include <iostream>
#include <vector>

int main() {

	const float INF = std::numeric_limits<float>::infinity();

	// Create an empty grid
	int m = 10;
	int n = 16;
	float* weights = new float[m*n];
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			weights[i*n+j] = 1.0;
		}
	}

	// Add some obstacles
	weights[4 * n + 3] = INF;
	weights[4 * n + 4] = INF;
	weights[4 * n + 5] = INF;

	// Path finding
	PathFinder pf = PathFinder(weights, m, n);

	Node start = Node(0, 0);
	Node end = Node(8, 12);
	const char* heuristic_name = "l2-norm";
	std::vector<Node> path = pf.findPath(start, end, heuristic_name, true);

	std::cout << "Path found:" << std::endl;
	for (unsigned int i=0; i < path.size(); ++i){
		std::cout << "(" << path[i].i << "," << path[i].j << ")" << std::endl;
	}
	return 0;
}
