# Copyright (c) 2017, IGLU consortium
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

from panda3d.core import NodePath

class World(object):
    
    def step(self, dt):
        raise NotImplementedError()

class Scene(object):
    
    def __init__(self, sceneId=None):
        self.scene = NodePath('scene')
        if sceneId is not None:
            self.scene.setTag('scene-id', str(sceneId))
        self.sceneId = sceneId
        agents = self.scene.attachNewNode('agents')
        self.agents = [agents.attachNewNode('agent-0'),]
        self.agents[0].setTag('agent-id', 'agent-0')
        
        self.worlds = dict()

    def getAllHouses(self):
        return self.scene.findAllMatches('**/house-*')

    def getAllRooms(self):
        return self.scene.findAllMatches('**/room-*')

    def getAllObjects(self):
        return self.scene.findAllMatches('**/object-*')

    def getAllAgents(self):
        return self.scene.findAllMatches('**/agents*')

    def getTotalNbHouses(self):
        nodepaths = self.getAllHouses()
        return len(nodepaths) if nodepaths else 0

    def getTotalNbRooms(self):
        nodepaths = self.getAllRooms()
        return len(nodepaths) if nodepaths else 0

    def getTotalNbObjects(self):
        nodepaths = self.getAllObjects()
        return len(nodepaths) if nodepaths else 0

    def getTotalNbAgents(self):
        nodepaths = self.getAllAgents()
        return len(nodepaths) if nodepaths else 0

    def hasGround(self):
        if self.scene.findAllMatches('**/ground*'):
            return True
        return False
    
    def __str__(self):
        return "Scene: %d houses, %d rooms, %d objects, %d agents" % ( self.getTotalNbHouses(),
                                                                self.getTotalNbRooms(),
                                                                self.getTotalNbObjects(),
                                                                self.getTotalNbAgents())
     
    __repr__ = __str__
    