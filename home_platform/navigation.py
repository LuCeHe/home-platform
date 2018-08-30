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

from __future__ import print_function

import os
import logging
import heapq
import networkx as nx
import numpy as np
import sknw

from skimage.morphology import skeletonize
from scipy import ndimage

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from panda3d.core import LVector3f, TransformState, NodePath

from home_platform.rendering import get3DTrianglesFromModel
from home_platform.suncg import loadModel
from skimage.morphology._skeletonize import medial_axis

logger = logging.getLogger(__name__)

MODEL_DATA_DIR = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "..", "data", "models")


def canvas2image(canvas):
    canvas.draw()       # draw the canvas, cache the renderer
    w, h = canvas.get_width_height()
    image = np.frombuffer(canvas.tostring_rgb(),
                          dtype='uint8').reshape((h, w, 3))
    image = image.astype(np.float) / 255.0
    return image


def getApproximationForModel(model, mode='box'):

    if mode == 'mesh':
        transform = model.getNetTransform()
        approxModel = model.copyTo(model.getParent())
        approxModel.detachNode()
        approxModel.setTransform(transform)

    elif mode == 'box':
        # Bounding box approximation
        # FIXME: taking the tight bounds after the transform does not fit well models
        #        that are rotated (e.g. diagonal orientation).
        minRefBounds, maxRefBounds = model.getTightBounds()
        refDims = maxRefBounds - minRefBounds
        refPos = model.getPos()
        refCenter = minRefBounds + (maxRefBounds - minRefBounds) / 2.0
        refDeltaCenter = refCenter - refPos

        approxModel = loadModel(os.path.join(MODEL_DATA_DIR, 'cube.egg'))

        # Rescale the cube model to match the bounding box of the original
        # model
        minBounds, maxBounds = approxModel.getTightBounds()
        dims = maxBounds - minBounds
        pos = approxModel.getPos()
        center = minBounds + (maxBounds - minBounds) / 2.0
        deltaCenter = center - pos

        position = refPos + refDeltaCenter - deltaCenter
        scale = LVector3f(refDims.x / dims.x, refDims.y /
                          dims.y, refDims.z / dims.z)

        # NOTE: remove the local transform used to center the model
        transform = model.getNetTransform().compose(model.getTransform().getInverse())
        transform = transform.compose(TransformState.makePosHprScale(position,
                                                                     LVector3f(
                                                                         0, 0, 0),
                                                                     scale))
        approxModel.setTransform(transform)
    else:
        raise Exception(
            'Unknown mode type for object shape: %s' % (mode))

    approxModel.setName(model.getName())

    return approxModel


def addModelTriangles(ax, model, invert=False, zlim=None):

    bounds = model.getTightBounds()
    if bounds is not None:
        minPt, maxPt = bounds
        zmin = minPt.z
        zmax = maxPt.z
        if zlim is None or (not (zmax <= zlim[0] or zmin >= zlim[1])):

            triangles = get3DTrianglesFromModel(model)

            x = triangles[:, :, 0].ravel()
            y = triangles[:, :, 1].ravel()
            indices = np.arange(
                3 * len(triangles)).reshape((len(triangles), 3))

            if invert:
                facecolors = np.ones((len(indices),))
                ax.tripcolor(x, y, indices, facecolors=facecolors,
                             edgecolors='white', vmin=0.0, vmax=1.0, cmap='gray')
            else:
                facecolors = np.zeros((len(indices),))
                ax.tripcolor(x, y, indices, facecolors=facecolors,
                             edgecolors='k', vmin=0.0, vmax=1.0, cmap='gray')


# Adapted from:
# https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
def set_axes_equal(ax):

    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)

    plot_radius = 0.5 * max([x_range, y_range])
    ax.set_xlim([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim([y_middle - plot_radius, y_middle + plot_radius])


def getRegionLabeledOccupacyMap(occupacyMap):

    occupacyMap = 1.0 - occupacyMap
    labeledOccupacyMap = np.atleast_2d(occupacyMap).astype(np.int)
    labeledOccupacyMap[labeledOccupacyMap != 0] = -1

    curRegionId = 1
    while np.count_nonzero(labeledOccupacyMap == 0) > 0:

        # Apply Grassfire algorithm (also known as Wavefront or Brushfire
        # algorithm)
        heap = []
        heapq.heapify(heap)
        visited = set()

        # Select a start point that is not yet labelled, add the heap queue
        start = tuple(np.argwhere(labeledOccupacyMap == 0)[0])
        heapq.heappush(heap, start)
        visited.add(start)

        while len(heap) > 0:
            # Get cell from heap queue, assign to current region and add to
            # visited set
            i, j = heapq.heappop(heap)
            labeledOccupacyMap[i, j] = curRegionId

            # Add all 4 neighbors to heap queue, if not occupied
            for ai, aj in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:

                # Check bounds
                if (ai >= 0 and ai < labeledOccupacyMap.shape[0] and aj >= 0 and aj < labeledOccupacyMap.shape[1]):
                    # Check occupancy and redundancy
                    if labeledOccupacyMap[ai, aj] == 0 and (ai, aj) not in visited:
                        heapq.heappush(heap, (ai, aj))
                        visited.add((ai, aj))

        curRegionId += 1

    nbRegions = curRegionId - 1

    assert np.all(labeledOccupacyMap != 0)
    logger.debug('Number of regions found in occupancy map: %d' % (nbRegions))

    return labeledOccupacyMap


def getModelBaseDimensions(model):

    origTransform = model.getNetTransform()

    model = model.copyTo(model.getParent())
    model.detachNode()
    model.setTransform(TransformState.makePosHprScale(origTransform.getPos(),
                                                      LVector3f(0, 0, 0),
                                                      origTransform.getScale()))
    minRefBounds, maxRefBounds = model.getTightBounds()
    model.removeNode()

    refDims = maxRefBounds - minRefBounds
    width, depth, heigth = refDims.x, refDims.y, refDims.z
    return width, depth, heigth


def getRandom2dMapCoordinates(occupancyMap):
    validIndices = np.argwhere(occupancyMap == 1.0)
    if len(validIndices) == 0:
        raise Exception('Occupancy map has no valid regions')
    idx = np.random.randint(0, len(validIndices))

    # NOTE: occupancy map is an image and thus the Y-axis is the first dimension
    xi = int(validIndices[idx][1])
    yi = int(validIndices[idx][0])

    return (yi, xi)


class NavigationHelper(object):

    closedStandardDoorModelIds = [
        # NOTE: '326' is a double glass door
        '326'
    ]

    openedStandardDoorModelIds = [
        '122', '133', '214', '246', '247', '327', '331', '73', '757', '758', '759', '760', '761', '762',
        '763', '764', '768', '769', '770', 's__1762', 's__1763', 's__1764', 's__1765', 's__1766',
        's__1767', 's__1768', 's__1769', 's__1770', 's__1771', 's__1772', 's__1773',
    ]
    openedThinDoorModelIds = [
        # NOTE: this list also covers archs
        '756', '778', '779', '780',
    ]
    openedGarageDoorModelIds = [
        '361', '765', '771',
    ]

    def __init__(self, scene):
        self.scene = scene

    def _isDoor(self, modelId):
        return (modelId in self.openedStandardDoorModelIds or
                modelId in self.openedThinDoorModelIds or
                modelId in self.openedGarageDoorModelIds)

    def _getFloorReferenceZ(self, level):
        # Loop for all floors in the scene:
        zmax = None
        for model in self.scene.scene.findAllMatches('**/level-%d/**/layouts/object-*/+ModelNode' % level):
            modelId = model.getNetTag('model-id')
            if not modelId.endswith('f'):
                continue

            bounds = model.getTightBounds()
            if bounds is not None:
                maxPt = bounds[1]
                if zmax is None or maxPt.z > zmax:
                    zmax = maxPt.z
        return zmax

    def calculateFloorMap(self, resolution=0.01, level=0, xlim=None, ylim=None, ignoreGround=False, roomIds=None):

        figSize = (10, 10)
        fig = Figure(figsize=figSize,
                     dpi=100, frameon=False)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111, aspect='equal')
        fig.subplots_adjust(left=0, bottom=0, right=1,
                            top=1, wspace=0, hspace=0)
        ax.axis('off')
        ax.set_aspect('equal')

        if roomIds is not None:
            models = []
            for roomId in roomIds:
                models.extend([model for model in self.scene.scene.findAllMatches(
                    '**/level-%d/room-%s/layouts/object-*/+ModelNode' % (level, roomId))])
        else:
            models = [model for model in self.scene.scene.findAllMatches(
                '**/level-%d/**/layouts/object-*/+ModelNode' % level)]

        # Loop for all floors in the scene:
        for model in models:
            modelId = model.getNetTag('model-id')

            if not modelId.endswith('f') or (ignoreGround and 'gd' in modelId):
                continue

            addModelTriangles(ax, model)

        if xlim is not None and ylim is not None:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        else:
            ax.autoscale(True)
            set_axes_equal(ax)

        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        xrange = xlim[1] - xlim[0]
        yrange = ylim[1] - ylim[0]
        assert np.allclose(xrange, yrange, atol=1e-6)
        dpi = (xrange / resolution) / figSize[0]
        fig.set_dpi(dpi)

        floorMap = canvas2image(canvas)
        plt.close(fig)

        # RGB to binary
        floorMap = np.round(np.mean(floorMap[:, :], axis=-1))

        # NOTE: Filter out small gaps that can exist between rooms
        floorMap = ndimage.gaussian_filter(floorMap, sigma=(1, 1), order=0)
        floorMap = np.round(floorMap)

        # NOTE: inverse image so that floor areas are shown in white
        floorMap = 1.0 - floorMap

        return floorMap, xlim, ylim

    def calculateObstacleMap(self, resolution=0.1, level=0, zlim=(0.15, 1.50), xlim=None, ylim=None, roomIds=None, layoutOnly=False):

        figSize = (10, 10)
        fig = Figure(figsize=figSize,
                     dpi=100, frameon=False)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111, aspect='equal')
        fig.subplots_adjust(left=0, bottom=0, right=1,
                            top=1, wspace=0, hspace=0)
        ax.axis('off')
        ax.set_aspect('equal')

        floorZ = self._getFloorReferenceZ(level)
        zlim = np.array(zlim) + floorZ

        if roomIds is not None:
            layoutModels = []
            objModels = []
            for roomId in roomIds:
                layoutModels.extend([model for model in self.scene.scene.findAllMatches(
                    '**/level-%d/room-%s/layouts/object-*/+ModelNode' % (level, roomId))])
                objModels.extend([model for model in self.scene.scene.findAllMatches(
                    '**/level-%d/room-%s/objects/object-*/+ModelNode' % (level, roomId))])
        else:
            layoutModels = [model for model in self.scene.scene.findAllMatches(
                '**/level-%d/**/layouts/object-*/+ModelNode' % level)]
            objModels = [model for model in self.scene.scene.findAllMatches(
                '**/level-%d/**/objects/object-*/+ModelNode' % level)]

        # Loop for all walls in the scene:
        for model in layoutModels:
            modelId = model.getNetTag('model-id')

            if not modelId.endswith('w'):
                continue

            addModelTriangles(ax, model, zlim=zlim)

        # Loop for all doors in the scene:
        for model in objModels:

            modelId = model.getNetTag('model-id')
            if self._isDoor(modelId):

                if modelId in self.openedStandardDoorModelIds:
                    # Shift the model a little more to the wall
                    transform = TransformState.makePos(
                        LVector3f(0.0, -0.10, 0.0))
                    # Reduce width by 25% not to mess with close corner walls
                    transform = transform.compose(
                        TransformState.makeScale(LVector3f(0.75, 1.0, 1.0)))
                elif modelId in self.openedThinDoorModelIds:
                    # Rescale the model to be able to cover the entire depth of walls
                    # Reduce width by 25% not to mess with close corner walls
                    transform = TransformState.makeScale(
                        LVector3f(0.75, 4.0, 1.0))
                elif modelId in self.openedGarageDoorModelIds:
                    # Shift the model a little more to the wall
                    transform = TransformState.makePos(
                        LVector3f(0.0, 0.10, 0.0))
                    # Reduce width by 10% not to mess with close corner walls
                    transform = transform.compose(
                        TransformState.makeScale(LVector3f(0.90, 1.0, 1.0)))
                else:
                    raise Exception('Unsupported model id: %s' % (modelId))

                # TODO: would be more efficient if it needed not copying the
                # model
                parentNp = NodePath('tmp-objnode')
                parentNp.setTransform(model.getParent().getNetTransform())
                midNp = parentNp.attachNewNode('tmp-transform')
                midNp.setTransform(transform)
                model = model.copyTo(midNp)
                approxModel = getApproximationForModel(model, mode='box')
                addModelTriangles(ax, approxModel, invert=True, zlim=zlim)
                approxModel.removeNode()
                midNp.removeNode()
                parentNp.removeNode()

        # Loop for all objects in the scene:
        if not layoutOnly:
            for model in objModels:

                modelId = model.getNetTag('model-id')
                if self._isDoor(modelId):
                    continue

                approxModel = getApproximationForModel(model, mode='box')
                addModelTriangles(ax, approxModel, zlim=zlim)

        if xlim is not None and ylim is not None:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        else:
            ax.autoscale(True)
            set_axes_equal(ax)

        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        xrange = xlim[1] - xlim[0]
        yrange = ylim[1] - ylim[0]
        assert np.allclose(xrange, yrange, atol=1e-6)
        dpi = (xrange / resolution) / figSize[0]
        fig.set_dpi(dpi)

        obstacleMap = canvas2image(canvas)
        plt.close(fig)

        # RGB to binary
        obstacleMap = np.round(np.mean(obstacleMap[:, :], axis=-1))

        # NOTE: inverse image so that obstacle areas are shown in white
        obstacleMap = 1.0 - obstacleMap

        return obstacleMap, xlim, ylim

    def calculateWallMap(self, resolution=0.1, level=0, zlim=(0.15, 1.50), xlim=None, ylim=None, roomIds=None):

        figSize = (10, 10)
        fig = Figure(figsize=figSize,
                     dpi=100, frameon=False)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111, aspect='equal')
        fig.subplots_adjust(left=0, bottom=0, right=1,
                            top=1, wspace=0, hspace=0)
        ax.axis('off')
        ax.set_aspect('equal')

        floorZ = self._getFloorReferenceZ(level)
        zlim = np.array(zlim) + floorZ

        if roomIds is not None:
            layoutModels = []
            for roomId in roomIds:
                layoutModels.extend([model for model in self.scene.scene.findAllMatches(
                    '**/level-%d/room-%s/layouts/object-*/+ModelNode' % (level, roomId))])
        else:
            layoutModels = [model for model in self.scene.scene.findAllMatches(
                '**/level-%d/**/layouts/object-*/+ModelNode' % level)]

        # Loop for all walls in the scene:
        for model in layoutModels:
            modelId = model.getNetTag('model-id')

            if not modelId.endswith('w'):
                continue

            addModelTriangles(ax, model, zlim=zlim)

        if xlim is not None and ylim is not None:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        else:
            ax.autoscale(True)
            set_axes_equal(ax)

        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        xrange = xlim[1] - xlim[0]
        yrange = ylim[1] - ylim[0]
        assert np.allclose(xrange, yrange, atol=1e-6)
        dpi = (xrange / resolution) / figSize[0]
        fig.set_dpi(dpi)

        wallMap = canvas2image(canvas)
        plt.close(fig)

        # RGB to binary
        wallMap = np.round(np.mean(wallMap[:, :], axis=-1))

        # NOTE: inverse image so that obstacle areas are shown in white
        wallMap = 1.0 - wallMap

        return wallMap, xlim, ylim

    def calculateRoomMaps(self, resolution=0.1, level=0, zlim=(0.15, 1.50), isBoundedByFloorMap=True, roomIds=None):

        floorZ = self._getFloorReferenceZ(level)
        zlim = np.array(zlim) + floorZ

        if isBoundedByFloorMap:
            _, xlim, ylim = self.calculateFloorMap(
                resolution, level, roomIds)
        else:
            _, xlim, ylim = self.calculateObstacleMap(
                resolution, level, zlim, roomIds)

        roomMaps = dict()
        if roomIds is None:
            roomIds = [roomNp.getTag(
                'room-id') for roomNp in self.scene.scene.findAllMatches('**/level-%d/room-*' % (level))]
        for roomId in roomIds:
            roomMap, _, _ = self.calculateFloorMap(
                resolution, level, xlim, ylim, roomIds=[roomId])

            roomMaps[roomId] = roomMap

        return roomMaps, xlim, ylim

    def calculateOccupancyMap(self, resolution=0.1, level=0, zlim=(0.15, 1.50), isBoundedByFloorMap=True, roomIds=None, layoutOnly=False):

        floorZ = self._getFloorReferenceZ(level)
        zlim = np.array(zlim) + floorZ

        if isBoundedByFloorMap:
            _, xlim, ylim = self.calculateFloorMap(
                resolution, level)
            floorOccupancyMap, xlim, ylim = self.calculateFloorMap(
                resolution, level, xlim, ylim, roomIds=roomIds)
            obstacleMap, _, _ = self.calculateObstacleMap(
                resolution, level, zlim, xlim, ylim, roomIds=roomIds, layoutOnly=layoutOnly)
        else:
            obstacleMap, xlim, ylim = self.calculateObstacleMap(
                resolution, level, zlim, roomIds=roomIds, layoutOnly=layoutOnly)
            floorOccupancyMap, _, _ = self.calculateFloorMap(
                resolution, level, xlim, ylim, roomIds=roomIds)

        occupancyMap = floorOccupancyMap * (1.0 - obstacleMap)

        return occupancyMap, xlim, ylim

    def calculateNavigationGraph(self, resolution=0.1, level=0, safetyMarginEdges=0.1, algorithm='medial-axis',
                                 shortLeafEdgesThresh=0.5, squishedLeafNodesThresh=0.15, redundantNodesThresh=0.25, longEdgesThreshold=0.5, doRemoveInaccessibleNodes=True,
                                 zlim=(0.15, 1.50), isBoundedByFloorMap=True, roomIds=None):
        occupancyMap, xlim, ylim = self.calculateOccupancyMap(
            resolution, level, zlim, isBoundedByFloorMap, roomIds)

        if algorithm == 'medial-axis':
            skeletonMap = medial_axis(occupancyMap)
        elif algorithm == 'skeleton':
            skeletonMap = skeletonize(occupancyMap)
        else:
            raise Exception('Unsupported algorithm: %s' % (algorithm))

        distOccupancyMap = ndimage.distance_transform_edt(occupancyMap)

        factorX = occupancyMap.shape[0] / \
            (xlim[1] - xlim[0])  # pixel per meter
        factorY = occupancyMap.shape[1] / \
            (ylim[1] - ylim[0])  # pixel per meter
        assert np.allclose(factorX, factorY, atol=1e-6)
        nbSafetyPixels = int(safetyMarginEdges * factorX)
        skeletonMap[distOccupancyMap < nbSafetyPixels] = 0.0

        graph = sknw.build_sknw(skeletonMap)

        graph = getLargestGraphOnly(graph)

        # FIXME: should probably run a loop here for several iterations
        graph = removeSelfEdges(graph)
        graph = removeShortLeafEdges(
            graph, threshold=shortLeafEdgesThresh * factorX)
        graph = removeSquishedLeafNodes(
            graph, distOccupancyMap, threshold=squishedLeafNodesThresh * factorX)
        graph = subdiviseLongEdges(
            graph, threshold=longEdgesThreshold * factorX)
        graph = removeRedundantNodes(
            graph, threshold=redundantNodesThresh * factorX)
        if doRemoveInaccessibleNodes:
            graph = removeInaccessibleNodes(graph, occupancyMap)

        graph = getLargestGraphOnly(graph)

        graph = NavigationGraph.fromNx(graph, factorX, xlim, ylim)

        return graph, occupancyMap, xlim, ylim


def getLargestGraphOnly(graph):

    # extract subgraphs
    subGraphs = [graph.subgraph(c).copy()
                 for c in nx.connected_components(graph)]
    maxNbNodes = subGraphs[0].number_of_nodes()
    bestGraph = subGraphs[0]
    for sg in subGraphs[1:]:
        if sg.number_of_nodes() >= maxNbNodes:
            maxNbNodes = sg.number_of_nodes()
            bestGraph = sg
    graph = bestGraph

    logger.debug('Largest graph found has %d nodes' %
                 (graph.number_of_nodes()))
    return graph


def removeSelfEdges(graph):

    edgesToRemove = []
    for s in graph.nodes():
        for e in graph.neighbors(s):
            if s == e:
                edgesToRemove.append((s, e))

    if len(edgesToRemove) > 0:
        graph.remove_edges_from(edgesToRemove)

    logger.debug('Total number of edges removed (self-loop): %d' % (
        len(edgesToRemove)))
    return graph


def subdiviseLongEdges(graph, threshold):

    # Split edges that are too long
    edgesToSplit = []
    for (s, e) in graph.edges():
        pts = graph[s][e]['pts']
        w = np.sum(
            np.sqrt(np.sum(np.square(np.diff(pts.astype(np.float), axis=0)), axis=-1)))

        if w > threshold and threshold > 0.0:
            edgesToSplit.append((s, e))

    availableNodeIdx = np.max([i for i in graph.nodes()]) + 1
    for (s, e) in edgesToSplit:

        pts = graph[s][e]['pts']
        w = np.sum(
            np.sqrt(np.sum(np.square(np.diff(pts.astype(np.float), axis=0)), axis=-1)))

        # Remove edge
        graph.remove_edge(s, e)

        # Split the path into equal-sized segment no longer than the threshold
        subSegments = np.array_split(pts, int(np.ceil(w / threshold)))
        assert len(subSegments) > 1
        m = None
        for i, subSegment in enumerate(subSegments):
            assert len(subSegment) > 1

            if i == 0:
                ss = s
                m = availableNodeIdx
                psm = subSegment[-1]
                availableNodeIdx += 1
                graph.add_node(m, o=psm)

            elif i == len(subSegments) - 1:
                ss = m
                m = e

            else:
                ss = m
                m = availableNodeIdx
                psm = subSegment[-1]
                availableNodeIdx += 1
                graph.add_node(m, o=psm)

            assert ss != m
            ps = np.array(graph.node[ss]['o'])
            dist = np.sum(
                np.sqrt(np.sum(np.square(np.diff(subSegment.astype(np.float), axis=0)), axis=-1)))
            graph.add_edge(ss, m, pts=np.array(
                [ps, psm]), weight=dist)

    logger.debug('Total number of edges split (too long): %d' % (
        len(edgesToSplit)))

    return graph


def removeShortLeafEdges(graph, threshold):

    # Remove edges that are too short
    nbTotalNodesRemoved = 0
    while True:
        # draw edges by pts
        toRemoveNodes = []
        for (s, e) in graph.edges():
            w = graph[s][e]['weight']
            ns = len([n for n in graph.neighbors(s)])
            ne = len([n for n in graph.neighbors(e)])
            n = min(ns, ne)

            if n < 2 and w < threshold:
                if ns <= 1:
                    toRemoveNodes.append(s)
                if ne <= 1:
                    toRemoveNodes.append(e)

        if len(toRemoveNodes) > 0:
            graph.remove_nodes_from(toRemoveNodes)
            nbTotalNodesRemoved += len(toRemoveNodes)
        else:
            break

    logger.debug('Total number of nodes removed (leaf edges too short): %d' % (
        nbTotalNodesRemoved))

    return graph


def removeSquishedLeafNodes(graph, distOccupancyMap, threshold):

    # Remove any leaf node that is too squished
    nbTotalNodesRemoved = 0
    while True:
        toRemoveNodes = []
        node = graph.node
        for i in graph.nodes():
            ns = len([n for n in graph.neighbors(i)])
            if ns <= 1:
                ps = np.round(np.array(node[i]['o'])).astype(np.int)
                dist = distOccupancyMap[ps[0], ps[1]]
                if dist <= threshold:
                    toRemoveNodes.append(i)

        if len(toRemoveNodes) > 0:
            graph.remove_nodes_from(toRemoveNodes)
            nbTotalNodesRemoved += len(toRemoveNodes)
        else:
            break

    logger.debug('Total number of nodes removed (leaf nodes too squished): %d' % (
        nbTotalNodesRemoved))

    return graph


def removeRedundantNodes(graph, threshold):

    # Merge nodes that are too close
    nbTotalNodesRemoved = 0
    while True:

        # Apply Grassfire algorithm (also known as Wavefront or Brushfire
        # algorithm)
        heap = []
        heapq.heapify(heap)
        visited = set()

        # Select a start node in the graph
        node = graph.node
        start = list(graph.nodes())[0]
        heapq.heappush(heap, start)
        visited.add(start)

        mustBreak = False
        redundantNodes = tuple()
        availableNodeIdx = np.max([i for i in graph.nodes()]) + 1
        while len(heap) > 0 and not mustBreak:
            # Get cell from heap queue, assign to current region and add to
            # visited set
            i = heapq.heappop(heap)

            # Add all neighbors to heap queue, if not visited
            for j in graph.neighbors(i):
                assert i != j

                # Check distance between nodes
                psi = np.array(node[i]['o'])
                psj = np.array(node[j]['o'])
                dist = np.sqrt(np.sum(np.square((psi - psj).astype(np.float))))

                if dist < threshold:
                    redundantNodes = (i, j)
                    mustBreak = True
                    break

                if j not in visited:
                    heapq.heappush(heap, j)
                    visited.add(j)

        if len(redundantNodes) > 0:
            i, j = redundantNodes

            psi = np.array(graph.node[i]['o'])
            psj = np.array(graph.node[j]['o'])
            psm = (psi + psj) // 2
            m = availableNodeIdx

            graph.add_node(m, o=psm)

            # NOTE: remove the edge between the two nodes before rewiring
            graph.remove_edge(i, j)

            for k in graph.neighbors(i):
                assert i != k
                assert k != m
                psk = np.array(graph.node[k]['o'])
                dist = np.sqrt(
                    np.sum(np.square((psk - psm).astype(np.float))))
                graph.add_edge(k, m, pts=np.array(
                    [psk, psm]), weight=dist)

            for k in graph.neighbors(j):
                assert j != k
                assert k != m
                psk = np.array(graph.node[k]['o'])
                dist = np.sqrt(
                    np.sum(np.square((psk - psm).astype(np.float))))
                graph.add_edge(m, k, pts=np.array(
                    [psm, psk]), weight=dist)

            graph.remove_nodes_from([i, j])
            nbTotalNodesRemoved += 1
        else:
            break

    logger.debug('Total number of nodes removed (redundant): %d' %
                 (nbTotalNodesRemoved))

    return graph


def removeInaccessibleNodes(graph, occupancyMap):

    # Remove nodes that are not accessible using a direct path
    toRemoveNodes = []
    for i in graph.nodes():
        ps = np.array(graph.node[i]['o'])
        for k in graph.neighbors(i):
            psk = np.array(graph.node[k]['o'])

            # Adapted from:
            # https://stackoverflow.com/questions/7878398/how-to-extract-an-arbitrary-line-of-values-from-a-numpy-array
            x0, y0 = ps.astype(np.float)
            x1, y1 = psk.astype(np.float)
            length = int(np.hypot(x1 - x0, y1 - y0))
            x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)

            # Extract the values along the line
            v = occupancyMap[x.astype(np.int), y.astype(np.int)]
            if not np.all(np.equal(v, 1.0)):
                toRemoveNodes.append(k)

    if len(toRemoveNodes) > 0:
        graph.remove_nodes_from(toRemoveNodes)
    nbTotalNodesRemoved = len(toRemoveNodes)

    logger.debug('Total number of nodes removed (not accessible): %d' %
                 (nbTotalNodesRemoved))

    return graph


class NavigationGraph(object):

    def __init__(self, nodes, connectivity):
        self.nodes = nodes
        self.connectivity = connectivity

    def toNx(self):
        graph = nx.Graph()

        # Add each node to the graph, with position information
        for i, position in enumerate(self.nodes):
            graph.add_node(i, position=position)

        # Add each edge to the graph
        for source, srcPos in enumerate(self.nodes):
            for target in self.connectivity[source]:
                targetPos = self.nodes[target]
                distance = np.sqrt(np.sum(np.square(targetPos - srcPos)))
                graph.add_edge(source, target, weight=distance)

        return graph

    @staticmethod
    def fromNx(graph, factor, xlim, ylim):

        di = dict()
        for i in graph.nodes():
            di[i] = len(di)

        nodes = []
        connectivity = []
        for i in graph.nodes():
            ps = np.array(graph.node[i]['o'], dtype=np.float)
            ps /= factor

            absPs = np.array(
                [xlim[0] + ps[1], ylim[1] - ps[0]], dtype=np.float)

            nodes.append(absPs)
            neighbors = []
            for k in graph.neighbors(i):
                neighbors.append(di[k])

            connectivity.append(neighbors)

        return NavigationGraph(nodes, connectivity)
