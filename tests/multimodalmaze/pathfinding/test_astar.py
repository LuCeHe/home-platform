# Copyright (c) 2018, Simon Brodeur
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  - Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import logging
import unittest
import numpy as np

from home_platform.pathfinding.astar import Node, PathFinder

logger = logging.getLogger(__name__)


class TestAstar(unittest.TestCase):

    def testFindPath(self):

        # Create an empty grid
        weights = np.ones((10, 16), dtype=np.float32)

        # Add some obstacles
        weights[4, 3:5] = np.inf

        # Path finding
        pf = PathFinder(weights)

        start = Node(0, 0)
        end = Node(8, 12)
        path = pf.findPath(start, end, 'l2-norm', use_diagonals=True)
        path = np.array([[node.i, node.j] for node in path])
        self.assertTrue(np.array_equal(path,
                                       np.array([(0, 0),
                                                 (1, 1),
                                                 (2, 2),
                                                 (3, 3),
                                                 (3, 4),
                                                 (4, 5),
                                                 (5, 6),
                                                 (6, 7),
                                                 (7, 8),
                                                 (8, 9),
                                                 (8, 10),
                                                 (8, 11),
                                                 (8, 12)]))
                        )

        start = Node(0, 0)
        end = Node(0, 8)
        path = pf.findPath(start, end, 'l2-norm', use_diagonals=False)
        path = np.array([[node.i, node.j] for node in path])
        self.assertTrue(np.array_equal(path,
                                       np.array([(0, 0),
                                                 (0, 1),
                                                 (0, 2),
                                                 (0, 3),
                                                 (0, 4),
                                                 (0, 5),
                                                 (0, 6),
                                                 (0, 7),
                                                 (0, 8)]))
                        )

        start = Node(0, 0)
        end = Node(42, 42)
        path = pf.findPath(start, end, 'l2-norm', use_diagonals=False)
        self.assertTrue(len(path) == 0)


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    unittest.main()
