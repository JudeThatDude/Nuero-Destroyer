import sys
import ctypes
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import FbxCommon

def initialize_pygame():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    glEnable(GL_DEPTH_TEST)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)
    
def load_fbx(filename):
    manager, scene = FbxCommon.InitializeSdkObjects()
    result = FbxCommon.LoadScene(manager, scene, filename)
    if not result:
        print("Failed to load FBX file")
        sys.exit(1)
    return scene

def draw_node(node):
    for i in range(node.GetChildCount()):
        child = node.GetChild(i)
        draw_node(child)
    mesh = node.GetMesh()
    if mesh:
        draw_mesh(mesh)

def draw_mesh(mesh):
    mesh.FBXDeformer = mesh.GetDeformerCount(FbxCommon.FbxDeformer.eSkin)
    if mesh.FBXDeformer == 0:
        return
    
    glBegin(GL_TRIANGLES)
    for i in range(mesh.GetPolygonCount()):
        for j in range(mesh.GetPolygonSize(i)):
            ctrl_point_index = mesh.GetPolygonVertex(i, j)
            ctrl_point = mesh.GetControlPointAt(ctrl_point_index)
            glVertex3f(ctrl_point[0], ctrl_point[1], ctrl_point[2])
    glEnd()

def main():
    filename = r"pdft-miku-with-bones\source\MikuFBX.fbx"
    initialize_pygame()
    scene = load_fbx(filename)
    
    clock = pygame.time.Clock()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == KEYDOWN:
                if event.key == K_LEFT:
                    glRotatef(-5, 0, 1, 0)
                if event.key == K_RIGHT:
                    glRotatef(5, 0, 1, 0)
                if event.key == K_UP:
                    glRotatef(-5, 1, 0, 0)
                if event.key == K_DOWN:
                    glRotatef(5, 1, 0, 0)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_node(scene.GetRootNode())
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
