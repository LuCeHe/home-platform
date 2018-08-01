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
import math
import logging

from panda3d.core import Point2D, Point3D, Vec3D, Vec4, GlobPattern, Filename,\
    CSZupRight, CSZupLeft, CSYupRight, CSYupLeft, NodePath
from panda3d.egg import EggTexture, EggMaterial, EggVertex, EggData, EggGroup, EggVertexPool, EggPolygon, EggLine,\
    loadEggData

logger = logging.getLogger(__name__)


def floats(float_list):
    """coerce a list of strings that represent floats into a list of floats"""
    return [float(number) for number in float_list]


def ints(int_list):
    """coerce a list of strings that represent integers into a list of integers"""
    return [int(number) for number in int_list]


class ObjMaterial:
    """a wavefront material

    Extended from: http://panda3d.org/phpbb2/viewtopic.php?t=3378
    licensed under WTFPL (http://sam.zoy.org/wtfpl/)
    """

    def __init__(self):
        self.filename = None
        self.name = "default"
        self.eggdiffusetexture = None
        self.eggmaterial = None
        self.attrib = {}
        self.attrib["Ns"] = 100.0
        self.attrib["d"] = 1.0
        self.attrib["illum"] = 2
        # "magenta"
        self.attrib["Kd"] = [1.0, 0.0, 1.0]
        self.attrib["Ka"] = [0.0, 0.0, 0.0]
        self.attrib["Ks"] = [0.0, 0.0, 0.0]
        self.attrib["Ke"] = [0.0, 0.0, 0.0]

    def put(self, key, value):
        self.attrib[key] = value
        return self

    def get(self, key):
        if key in self.attrib:
            return self.attrib[key]
        return None

    def has_key(self, key):
        return key in self.attrib

    def isTextured(self):
        # for k in ("map_Kd", "map_Bump", "map_Ks"):    <-- NOT YET
        if "map_Kd" in self.attrib:
            return True
        return False

    def getEggTexture(self):
        if self.eggdiffusetexture:
            return self.eggdiffusetexture
        if not self.isTextured():
            return None
        m = EggTexture(self.name + "_diffuse", self.get("map_Kd"))

        # Check if texture supports transparency based on extension name
        _, ext = os.path.splitext(self.attrib['map_Kd'])
        if ext.lower() == '.png':
            m.setFormat(EggTexture.FRgba)
        elif ext.lower() in ['.jpg', '.jpeg']:
            # NOTE: JPG format does not support transparency
            m.setFormat(EggTexture.FRgb)
        else:
            logger.debug('Object has texture with extension: %s' % str(ext))
            m.setFormat(EggTexture.FRgb)

        m.setMagfilter(EggTexture.FTLinearMipmapLinear)
        m.setMinfilter(EggTexture.FTLinearMipmapLinear)
        m.setWrapU(EggTexture.WMRepeat)
        m.setWrapV(EggTexture.WMRepeat)
        self.eggdiffusetexture = m
        return self.eggdiffusetexture

    def getEggMaterial(self):
        if self.eggmaterial:
            return self.eggmaterial
        m = EggMaterial(self.name + "_mat")
        # XXX TODO: add support for specular, and obey illum setting
        # XXX as best as we can
        rgb = self.get("Kd")
        d = self.get("d")
        if rgb is not None:
            m.setDiff(Vec4(rgb[0], rgb[1], rgb[2], d))
        rgb = self.get("Ka")
        if rgb is not None:
            m.setAmb(Vec4(rgb[0], rgb[1], rgb[2], 1.0))
        rgb = self.get("Ks")
        if rgb is not None:
            m.setSpec(Vec4(rgb[0], rgb[1], rgb[2], 1.0))
        ns = self.get("Ns")
        if ns is not None:
            m.setShininess(ns)
        self.eggmaterial = m
        return self.eggmaterial


class MtlFile:
    """an object representing all Wavefront materials in a .mtl file

    Extended from: http://panda3d.org/phpbb2/viewtopic.php?t=3378
    licensed under WTFPL (http://sam.zoy.org/wtfpl/)
    """

    def __init__(self, filename=None):
        self.filename = None
        self.materials = {}
        self.comments = {}
        if filename is not None:
            self.read(filename)

    def read(self, filename):
        self.filename = filename
        self.materials = {}
        self.comments = {}
        try:
            fd = open(filename)
        except Exception:
            return self
        linenumber = 0
        mat = None
        for line in fd.readlines():
            line = line.strip()
            linenumber = linenumber + 1
            if not line:
                continue
            if line[0] == '#':
                self.comments[linenumber] = line
                continue
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "newmtl":
                mat = ObjMaterial()
                mat.filename = filename
                mat.name = tokens[1]
                self.materials[mat.name] = mat
                continue
            if tokens[0] in ("Ns", "d", "Tr"):
                # "d factor" - specifies the dissovle for the current material,
                #              1.0 is full opaque
                # "Ns exponent" - specifies the specular exponent.  A high exponent
                #               results in a tight, concentrated highlight.
                mat.put(tokens[0], float(tokens[1]))
                continue
            if tokens[0] in ("illum"):
                # according to http://www.fileformat.info/format/material/
                # 0 = Color on and Ambient off
                # 1 = Color on and Ambient on
                # 2 = Highlight on
                # 3 = Reflection on and Ray trace on
                # 4 = Transparency: Glass on, Reflection: Ray trace on
                # 5 = Reflection: Fesnel on and Ray trace on
                # 6 = Transparency: Refraction on, Reflection: Fresnel off and Ray trace on
                # 7 = Transparency: Refraction on, Refelction: Fresnel on and Ray Trace on
                # 8 = Reflection on and Ray trace off
                # 9 = Transparency: Glass on, Reflection: Ray trace off
                # 10 = Casts shadows onto invisible surfaces
                mat.put(tokens[0], int(tokens[1]))
                continue
            if tokens[0] in ("Kd", "Ka", "Ks", "Ke"):
                mat.put(tokens[0], floats(tokens[1:]))
                continue
            if tokens[0] in ("map_Kd", "map_Bump", "map_Ks", "map_bump", "bump"):
                # Ultimate Unwrap 3D Pro emits these:
                # map_Kd == diffuse
                # map_Bump == bump
                # map_Ks == specular
                mat.put(tokens[0], pathify(tokens[1]))
                continue
            if tokens[0] in ("Ni"):
                # blender's .obj exporter can emit this "Ni 1.000000"
                mat.put(tokens[0], float(tokens[1]))
                continue
            logger.warning("file \"%s\": line %d: unrecognized: %s" %
                           (filename, linenumber, str(tokens)))
        fd.close()
        return self


class ObjFile:
    """a representation of a wavefront .obj file

    Extended from: http://panda3d.org/phpbb2/viewtopic.php?t=3378
    licensed under WTFPL (http://sam.zoy.org/wtfpl/)
    """

    def __init__(self, filename=None):
        self.filename = None
        self.objects = ["defaultobject"]
        self.groups = ["defaultgroup"]
        self.points = []
        self.uvs = []
        self.normals = []
        self.faces = []
        self.polylines = []
        self.matlibs = []
        self.materialsbyname = {}
        self.comments = {}
        self.currentobject = self.objects[0]
        self.currentgroup = self.groups[0]
        self.currentmaterial = None
        if filename is not None:
            self.read(filename)

    def read(self, filename):
        logger.debug("ObjFile.read: filename: %s" % (filename))
        self.filename = filename
        self.objects = ["defaultobject"]
        self.groups = ["defaultgroup"]
        self.points = []
        self.uvs = []
        self.normals = []
        self.faces = []
        self.polylines = []
        self.matlibs = []
        self.materialsbyname = {}
        self.comments = {}
        self.currentobject = self.objects[0]
        self.currentgroup = self.groups[0]
        self.currentmaterial = None
        try:
            fd = open(filename)
        except Exception:
            return self
        linenumber = 0
        for line in fd.readlines():
            line = line.strip()
            linenumber = linenumber + 1
            if not line:
                continue
            if line[0] == '#':
                self.comments[linenumber] = line
                continue
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "mtllib":
                logger.debug("mtllib: %s" % (str(tokens[1:])))

                mtlPath = tokens[1]
                if not os.path.exists(mtlPath):
                    # Check if relative path to obj
                    mtlPath = os.path.join(os.path.dirname(filename), mtlPath)
                    if not os.path.exists(mtlPath):
                        logger.warning(
                            'Could not find mtl file: %s' % (str(tokens[1])))

                mtllib = MtlFile(mtlPath)
                self.matlibs.append(mtllib)
                self.indexmaterials(mtllib)
                continue
            if tokens[0] == "g":
                self.__newgroup("".join(tokens[1:]))
                continue
            if tokens[0] == "o":
                self.__newobject("".join(tokens[1:]))
                continue
            if tokens[0] == "usemtl":
                self.__usematerial(tokens[1])
                continue
            if tokens[0] == "v":
                self.__newv(tokens[1:])
                continue
            if tokens[0] == "vn":
                self.__newnormal(tokens[1:])
                continue
            if tokens[0] == "vt":
                self.__newuv(tokens[1:])
                continue
            if tokens[0] == "f":
                self.__newface(tokens[1:])
                continue
            if tokens[0] == "s":
                # apparently, this enables/disables smoothing
                logger.debug("%s:%d: ignoring: %s" %
                             (filename, linenumber, str(tokens)))
                continue
            if tokens[0] == "l":
                self.__newpolyline(tokens[1:])
                continue
            logger.warning("%s:%d: unknown: %s" %
                           (filename, linenumber, str(tokens)))
        fd.close()
        return self

    def __vertlist(self, lst):
        res = []
        for vert in lst:
            vinfo = vert.split("/")
            vlen = len(vinfo)
            vertex = {'v': None, 'vt': None, 'vn': None}
            if vlen == 1:
                vertex['v'] = int(vinfo[0])
            elif vlen == 2:
                if vinfo[0] != '':
                    vertex['v'] = int(vinfo[0])
                if vinfo[1] != '':
                    vertex['vt'] = int(vinfo[1])
            elif vlen == 3:
                if vinfo[0] != '':
                    vertex['v'] = int(vinfo[0])
                if vinfo[1] != '':
                    vertex['vt'] = int(vinfo[1])
                if vinfo[2] != '':
                    vertex['vn'] = int(vinfo[2])
            else:
                raise Exception(str(res))
            res.append(vertex)
        return res

    def __enclose(self, lst):
        mdata = (self.currentobject, self.currentgroup, self.currentmaterial)
        return (lst, mdata)

    def __newpolyline(self, l):
        polyline = self.__vertlist(l)
        self.polylines.append(self.__enclose(polyline))
        return self

    def __newface(self, f):
        face = self.__vertlist(f)
        self.faces.append(self.__enclose(face))
        return self

    def __newuv(self, uv):
        self.uvs.append(floats(uv))
        return self

    def __newnormal(self, normal):
        self.normals.append(floats(normal))
        return self

    def __newv(self, v):
        # capture the current metadata with vertices
        vdata = floats(v)
        mdata = (self.currentobject, self.currentgroup, self.currentmaterial)
        vinfo = ([0 if math.isnan(vert) else vert for vert in vdata], mdata)
        self.points.append(vinfo)
        return self

    def indexmaterials(self, mtllib):
        # traverse the materials defined in mtllib, indexing
        # them by name.
        for mname in mtllib.materials:
            mobj = mtllib.materials[mname]
            self.materialsbyname[mobj.name] = mobj
        logger.debug("indexmaterials: %s materials: %s" % (str(mtllib.filename),
                                                           str(self.materialsbyname.keys())))
        return self

    def __closeobject(self):
        self.currentobject = "defaultobject"
        return self

    def __newobject(self, obj):
        self.__closeobject()
        self.currentobject = obj
        self.objects.append(obj)
        return self

    def __closegroup(self):
        self.currentgroup = "defaultgroup"
        return self

    def __newgroup(self, group):
        self.__closegroup()
        self.currentgroup = group
        self.groups.append(group)
        return self

    def __usematerial(self, material):
        if material in self.materialsbyname:
            self.currentmaterial = material
        else:
            logger.warning("warning: unknown material: %s" % (str(material)))
        return self

    def __itemsby(self, itemlist, objname, groupname):
        res = []
        for item in itemlist:
            _, mdata = item
            wobj, wgrp, _ = mdata
            if (wobj == objname) and (wgrp == groupname):
                res.append(item)
        return res

    def __facesby(self, objname, groupname):
        return self.__itemsby(self.faces, objname, groupname)

    def __linesby(self, objname, groupname):
        return self.__itemsby(self.polylines, objname, groupname)

    def __eggifyverts(self, eprim, evpool, vlist):
        for vertex in vlist:
            ixyz = vertex['v']
            vinfo = self.points[ixyz - 1]
            vxyz, _ = vinfo
            ev = EggVertex()
            ev.setPos(Point3D(vxyz[0], vxyz[1], vxyz[2]))
            iuv = vertex['vt']
            if iuv is not None:
                vuv = self.uvs[iuv - 1]
                ev.setUv(Point2D(vuv[0], vuv[1]))
            inormal = vertex['vn']
            if inormal is not None:
                vn = self.normals[inormal - 1]
                ev.setNormal(Vec3D(vn[0], vn[1], vn[2]))
            evpool.addVertex(ev)
            eprim.addVertex(ev)
        return self

    def __eggifymats(self, eprim, wmat):
        if wmat in self.materialsbyname:
            mtl = self.materialsbyname[wmat]
            if mtl.isTextured():
                eprim.setTexture(mtl.getEggTexture())
                # NOTE: it looks like you almost always want to setMaterial()
                #       for textured polys.... [continued below...]
                eprim.setMaterial(mtl.getEggMaterial())
            rgb = mtl.get("Kd")
            d = mtl.get("d")
            if rgb is not None:
                # ... and some untextured .obj's store the color of the
                # material # in the Kd settings...
                eprim.setColor(Vec4(rgb[0], rgb[1], rgb[2], d))
            # [continued...] but you may *not* always want to assign
            # materials to untextured polys...  hmmmm.
            if False:
                eprim.setMaterial(mtl.getEggMaterial())
        return self

    def __facestoegg(self, egg, objname, groupname):
        selectedfaces = self.__facesby(objname, groupname)
        if len(selectedfaces) == 0:
            return self
        eobj = EggGroup(objname)
        egg.addChild(eobj)
        egrp = EggGroup(groupname)
        eobj.addChild(egrp)
        evpool = EggVertexPool(groupname)
        egrp.addChild(evpool)
        for face in selectedfaces:
            vlist, mdata = face
            _, _, wmat = mdata
            epoly = EggPolygon()
            egrp.addChild(epoly)
            self.__eggifymats(epoly, wmat)
            self.__eggifyverts(epoly, evpool, vlist)
        return self

    def __polylinestoegg(self, egg, objname, groupname):
        selectedlines = self.__linesby(objname, groupname)
        if len(selectedlines) == 0:
            return self
        eobj = EggGroup(objname)
        egg.addChild(eobj)
        egrp = EggGroup(groupname)
        eobj.addChild(egrp)
        evpool = EggVertexPool(groupname)
        egrp.addChild(evpool)
        for line in selectedlines:
            vlist, mdata = line
            _, _, wmat = mdata
            eline = EggLine()
            egrp.addChild(eline)
            self.__eggifymats(eline, wmat)
            self.__eggifyverts(eline, evpool, vlist)
        return self

    def toEgg(self):
        # make a new egg
        egg = EggData()
        # convert polygon faces
        if len(self.faces) > 0:
            for objname in self.objects:
                for groupname in self.groups:
                    self.__facestoegg(egg, objname, groupname)
        # convert polylines
        if len(self.polylines) > 0:
            for objname in self.objects:
                for groupname in self.groups:
                    self.__polylinestoegg(egg, objname, groupname)
        return egg


def pathify(path):
    """
    Extended from: http://panda3d.org/phpbb2/viewtopic.php?t=3378
    licensed under WTFPL (http://sam.zoy.org/wtfpl/)
    """

    if os.path.isfile(path):
        return path
    # if it was written on win32, it may have \'s in it, and
    # also a full rather than relative pathname (Hexagon does this... ick)
    orig = path
    path = path.lower()
    path = path.replace("\\", "/")
    _, t = os.path.split(path)
    if os.path.isfile(t):
        return t
    logger.warning(
        "warning: can't make sense of this map file name: %s" % (str(orig)))
    return t


def obj2egg(infile, outfile, coordinateSystem='z-up', recomputeVertexNormals=False, recomputeTangentBinormal=False,
            recomputePolygonNormals=False, triangulatePolygons=False, degreeSmoothing=30.0):

    if coordinateSystem == 'z-up' or coordinateSystem == 'z-up-right':
        coordSys = CSZupRight
    elif coordinateSystem == 'z-up-left':
        coordSys = CSZupLeft
    elif coordinateSystem == 'y-up' or coordinateSystem == 'y-up-right':
        coordSys = CSYupRight
    elif coordinateSystem == 'y-up-left':
        coordSys = CSYupLeft
    else:
        raise Exception('Unsupported coordinate system: %s' %
                        (coordinateSystem))

    os.chdir(os.path.dirname(infile))
    obj = ObjFile(infile)
    egg = obj.toEgg()
    egg.setCoordinateSystem(coordSys)

    egg.removeUnusedVertices(GlobPattern(""))

    if recomputeVertexNormals:
        egg.recomputeVertexNormals(float(degreeSmoothing))

    if recomputeTangentBinormal:
        egg.recomputeTangentBinormal(GlobPattern(""))

    if recomputePolygonNormals:
        egg.recomputePolygonNormals()

    if triangulatePolygons:
        egg.triangulatePolygons(EggData.TConvex & EggData.TPolygon)

    egg.writeEgg(Filename(outfile))


def obj2bam(infile, outfile, coordinateSystem='z-up', recomputeVertexNormals=False, recomputeTangentBinormal=False,
            recomputePolygonNormals=False, triangulatePolygons=False, degreeSmoothing=30.0):

    if coordinateSystem == 'z-up' or coordinateSystem == 'z-up-right':
        coordSys = CSZupRight
    elif coordinateSystem == 'z-up-left':
        coordSys = CSZupLeft
    elif coordinateSystem == 'y-up' or coordinateSystem == 'y-up-right':
        coordSys = CSYupRight
    elif coordinateSystem == 'y-up-left':
        coordSys = CSYupLeft
    else:
        raise Exception('Unsupported coordinate system: %s' %
                        (coordinateSystem))

    os.chdir(os.path.dirname(infile))
    obj = ObjFile(infile)
    egg = obj.toEgg()
    egg.setCoordinateSystem(coordSys)

    egg.removeUnusedVertices(GlobPattern(""))

    if recomputeVertexNormals:
        egg.recomputeVertexNormals(float(degreeSmoothing))

    if recomputeTangentBinormal:
        egg.recomputeTangentBinormal(GlobPattern(""))

    if recomputePolygonNormals:
        egg.recomputePolygonNormals()

    if triangulatePolygons:
        egg.triangulatePolygons(EggData.TConvex & EggData.TPolygon)

    np = NodePath(loadEggData(egg))
    np.writeBamFile(outfile)
    np.removeNode()
