# -*- coding: utf-8 -*-
"""
    Copyright (c) 2021 Zhang Bei, rular099@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import pygame
from pygame.locals import *

import os
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import json
import quaternion
from scipy.spatial.transform import Rotation as R

class BoxRoll:
    def __init__(self,fname=None,useQuat=True):
        self.fname = fname
        if not fname is None:
            self.load_data(fname)
        else:
            self.gen_sample_sequence()
        self.useQuat=useQuat

    def load_data(self,fname):
        self.sequence = quaternion.as_quat_array(np.loadtxt(fname))

    def gen_sample_sequence(self,n=10000):
        data = np.zeros((n,4))
        data[:,0] = np.cos(np.linspace(0,np.pi,n))
        data[:,3] = np.sin(np.linspace(0,np.pi,n))
        self.sequence = quaternion.as_quat_array(data)

    def draw_cube(self, w, nx, ny, nz, useQuat=None):
        if useQuat is None:
            useQuat = self.useQuat

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0, 0.0, -7.0)

        self.drawText((-2.6, 1.6, 2), "Module to visualize quaternion or Euler angles data", 16)
        self.drawText((-2.6, -2, 2), "Press Escape to exit.", 16)

        # default view is from +z to -z, y pointing upward. We want look from
        # +x to -x, and +z points upward. So we need change z->x, x->y, y->z
        # if we rotate around x, infact we rotate around z, etc.
        if(useQuat):
            r = R.from_quat([nx,ny,nz,w])
            roll, pitch , yaw = r.as_euler('XYZ',degrees=True)
            self.drawText((-2.6, -1.8, 2), "Roll: %f, Pitch: %f, Yaw: %f" %(roll, pitch, yaw), 16)
            glRotatef(2 * np.arccos(w) * 180.00/np.pi, ny , nz , nx)
        else:
            yaw = nx
            pitch = ny
            roll = nz
            self.drawText((-2.6, -1.8, 2), "Yaw: %f, Pitch: %f, Roll: %f" %(yaw, pitch, roll), 16)
            glRotatef(roll, 0.00, 0.00, 1.00)
            glRotatef(pitch, 1.00, 0.00, 0.00)
            glRotatef(yaw, 0.00, 1.00, 0.00)

        glBegin(GL_QUADS)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(1.0, 0.2, -1.0)
        glVertex3f(-1.0, 0.2, -1.0)
        glVertex3f(-1.0, 0.2, 1.0)
        glVertex3f(1.0, 0.2, 1.0)

        glColor3f(1.0, 0.5, 0.0)
        glVertex3f(1.0, -0.2, 1.0)
        glVertex3f(-1.0, -0.2, 1.0)
        glVertex3f(-1.0, -0.2, -1.0)
        glVertex3f(1.0, -0.2, -1.0)

        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(1.0, 0.2, 1.0)
        glVertex3f(-1.0, 0.2, 1.0)
        glVertex3f(-1.0, -0.2, 1.0)
        glVertex3f(1.0, -0.2, 1.0)

        glColor3f(1.0, 1.0, 0.0)
        glVertex3f(1.0, -0.2, -1.0)
        glVertex3f(-1.0, -0.2, -1.0)
        glVertex3f(-1.0, 0.2, -1.0)
        glVertex3f(1.0, 0.2, -1.0)

        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(-1.0, 0.2, 1.0)
        glVertex3f(-1.0, 0.2, -1.0)
        glVertex3f(-1.0, -0.2, -1.0)
        glVertex3f(-1.0, -0.2, 1.0)

        glColor3f(1.0, 0.0, 1.0)
        glVertex3f(1.0, 0.2, -1.0)
        glVertex3f(1.0, 0.2, 1.0)
        glVertex3f(1.0, -0.2, 1.0)
        glVertex3f(1.0, -0.2, -1.0)
        glEnd()

    def drawText(self,position, textString, size):
        font = pygame.font.SysFont("Courier", size, True)
        textSurface = font.render(textString, True, (255, 255, 255, 255), (0, 0, 0, 255))
        textData = pygame.image.tostring(textSurface, "RGBA", True)
        glRasterPos3d(*position)
        glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)

    def show(self,sequence=None,useQuat=None,fps=50):
        if useQuat is None:
            useQuat = self.useQuat
        self._init_view()
        frames = 0
        ticks = pygame.time.get_ticks()
        fps_clock = pygame.time.Clock()
        for x in self.sequence:
            event = pygame.event.poll()
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                break
            if(useQuat):
                w,nx,ny,nz = x.components
                self.draw_cube(w, nx, ny, nz, useQuat)
            else:
                yaw,pitch,roll = x
                self.draw_cube(1, yaw, pitch, roll, useQuat)
            pygame.display.flip()
            fps_clock.tick(fps)
            frames += 1
        print("fps: %d" % ((frames*1000)/(pygame.time.get_ticks()-ticks)))

    def _init_view(self,eyepos=[30,0,30],center=[0,0,0],up_v=[0,0,-1]):
#        os.environ['SDL_VIDEO_WINDOW_POS'] = (10,10)
        pygame.init()
        width = 640
        height = 480
        pygame.display.set_mode((width,height), DOUBLEBUF|OPENGL)
        pygame.display.set_caption("IMU orientation visualization")
        # set size of window
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, 1.0*width/height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluPerspective(65, (width/height), 0.1, 50.0)

        glShadeModel(GL_SMOOTH)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)

if __name__ == '__main__':
    fname = './result_q.txt'
    box_roll = BoxRoll(fname)
    box_roll.show()
