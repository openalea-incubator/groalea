# TODO

# 1. Implement the full specification
# 2. Test all the cases with several examples
# 3. Implement a loop in OpenAlea
# 4. Use the PlantGL turtle

# 2.1. add a header
# 2.2. separate graph parsing and scenegraph generation
# 2.3. Error management
# 2.4. Documentation
# 2.5. Compute properties when it is possible (sphere, ...)
# 2.6. 2D draw of the graph


# 3. Add enum like FUNCTIONAL

from math import radians
from math import sqrt
from math import cos
from math import sin
from copy import deepcopy

import openalea.plantgl.all as pgl
from openalea.core.graph.property_graph import PropertyGraph

Vector3 = pgl.Vector3
Vector4 = pgl.Vector4
Color4Array = pgl.Color4Array



class RootedGraph(PropertyGraph):
    """ A general graph with a root vertex. """

    def _set_root(self, root):
        self._root = root

    def _get_root(self):
        return self._root

    root = property(_get_root, _set_root)




#######################################################################################

def rgb_color(index):
    """ Returns an RGB color from an index. """
    LUT_color = [(0, 0, 0),
                 (0, 0, 170),
                 (0, 170, 0),
                 (0, 170, 170),
                 (170, 0, 0),
                 (170, 0, 170),
                 (170, 85, 0),
                 (170, 170, 170),
                 (85, 85, 85),
                 (85, 85, 255),
                 (85, 255, 85),
                 (85, 255, 255),
                 (255, 85, 85),
                 (255, 85, 255),
                 (255, 255, 85),
                 (255, 255, 255), ]

    color = pgl.Color3(LUT_color[int(index)])
    return color


class TurtleState(object):
    """ Store the turtle state of each vertex. """
    DIAMETER = 0.1
    LENGTH = 0.1

    def __init__(self):
        self.diameter = -1
        self.diameter_add = 0
        self.diameter_mul = 1.
        self.set_diameter = False
        self.color = None
        self.length = -1.
        self.length_add = 0
        self.length_mul = 1.
        self.set_length = False

        self.tropism = 0.
        self.tropism_direction = None
        self.tropism_target = None

    def copy(self):
        return deepcopy(self)

    def combine(self, t):
        # copy to avoid side-effect
        cself = self.copy()
        if t.diameter != -1:
            cself.diameter = t.diameter
        if t.diameter_add != 0:
            if cself.diameter == -1:
                cself.diameter = self.DIAMETER
            cself.diameter += t.diameter_add
        if t.diameter_mul != 1:
            if cself.diameter == -1:
                cself.diameter = self.DIAMETER
            cself.diameter *= t.diameter_mul
        if t.set_diameter:
            if cself.diameter == -1 and t.diameter == -1:
                cself.diameter = self.DIAMETER
            elif cself.diameter == -1:
                cself.diameter = t.diameter

        if t.color:
            cself.color = t.color

        if t.length != -1:
            cself.length = t.length
        if t.length_add != 0:
            if cself.length == -1:
                cself.length = self.LENGTH
            cself.length += t.length_add
        if t.length_mul != 1:
            if cself.length == -1:
                cself.length = self.LENGTH
            cself.length *= t.length_mul
        if t.set_length:
            if cself.length == -1 and t.length == -1:
                cself.length = self.DIAMETER
            elif cself.length == -1:
                cself.length = t.length

        cself.tropism = t.tropism
        cself.tropism_direction = t.tropism_direction
        cself.tropism_target = t.tropism_target

        return cself
    def __eq__(self, other):
        """ Test for == operator """
        ok = ((self.diameter == other.diameter) and
              (self.diameter_add == other.diameter_add) and
              (self.diameter_mul == other.diameter_mul) and
              (self.set_diameter == other.set_diameter) and
              (self.color == other.color) and
              (self.length == other.length) and
              (self.length_add == other.length_add) and
              (self.length_mul == other.length_mul) and
              (self.set_length == other.set_length) and
              (self.tropism == other.tropism) and
              (self.tropism_direction == other.tropism_direction) and
              (self.tropism_target == other.tropism_target)
              )
        return ok

    def __nonzero__(self):
        return not self.__eq__(TurtleState())


class FunctionalGeometry(object):
    def __init__(self, function):
        self.function = function

    def __call__(self, turtle_state):
        return self.function(turtle_state)


#######################################


def is_matrix(shape):
    return type(shape) == pgl.Matrix4


def transform4(matrix, shape):
    """
    Return a shape transformed by a Matrix4.
    """
    if matrix == None:
        matrix = pgl.Matrix4()
    scale, (a, e, r), translation = matrix.getTransformation2()
    shape = pgl.Translated(translation,
                           pgl.Scaled(scale,
                                      pgl.EulerRotated(a, e, r,
                                                       shape)))
    return shape

##########################################################################
# Transformation functions

def orientation(v):
    area = 0.0
    # Compute the area (times 2) of the polygon
    for i in range(len(v)):
        area += v[i - 1][0] * v[i][1] - v[i - 1][1] * v[i][0]

    if area >= 0.0:
        return 0
    return 1

def project3Dto2D(p3list):
    v01 = Vector3((p3list[1][0] - p3list[0][0]), (p3list[1][1] - p3list[0][1]), (p3list[1][2] - p3list[0][2]))
    v12 = Vector3((p3list[2][0] - p3list[1][0]), (p3list[2][1] - p3list[1][1]), (p3list[2][2] - p3list[1][2]))
    vn = pgl.cross(v01, v12)

    p2s = []
    # cosTheta = A dot B/(|A|*|B|) => if A dot B ==0, then Theta == 90
    # if polygon not || y axis, project it to the y=0 plane
    if pgl.dot(vn, Vector3(0, 1, 0)) != 0:
        for i in range(len(p3list)):
            v = p3list[i][0], p3list[i][2]
            p2s.append(v)

    else:
        # if polygon || y axis and z axis (it will perpendicular x axis), project it to the x=0 plane
        # if polygon || y axis and x axis (it will perpendicular z axis), project it to the z=0 plane
        # if polygon || y axis, not || x and z (it will not perpendicular z and x
        # axis), project it to the z=0 plane (or x=0 plane)
        if pgl.dot(vn, Vector3(0, 0, 1)) == 0:
            for i in range(len(p3list)):
                v = p3list[i][1], p3list[i][2]
                p2s.append(v)

        else:
            for i in range(len(p3list)):
                v = p3list[i][0], p3list[i][1]
                p2s.append(v)
    return p2s


def determinant(p1, p2, p3):
    """ TODO : Use static method here
    """
    determ = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])

    if determ >= 0:
        return 1
    return 0


def no_interior(p1, p2, p3, v, poly_or):
    """ TODO : Use static method
    """
    for p in v:
        if p.values()[0] == p1 or p.values()[0] == p2 or p.values()[0] == p3:
            # Don't bother checking against yourself
            continue

        if ((determinant(p1, p2, p.values()[0]) == poly_or) or
            (determinant(p3, p1, p.values()[0]) == poly_or) or
            (determinant(p2, p3, p.values()[0]) == poly_or)):
            # This point is outside
            continue
        # The point is inside
        return False
    # No points inside this triangle
    return True

##########################################################################

class Frame(object):
    def __init__(self, matrix4):
        self.m = matrix4

    def grotation(self, strength):
        """ Doc """
        m = self.m
        t = m.getColumn(2)
        v0 = t.x
        v1 = t.y
        v2 = t.z
        q = 1 / sqrt(t.x * t.x + t.y * t.y + t.z * t.z)
        v0 *= q
        v1 *= q
        v2 *= q
        v02 = v0 * v0
        v12 = v1 * v1
        q = v02 + v12
        m00 = m11 = m22 = m10 = m20 = m01 = m21 = m02 = m12 = 0
        local_m = 0
        if (q < 1e-10) or (v2 * v2 > 0.99999):
            if v2 * (v2 - strength) < 0:
                m00 = m.getRow(0).x
                m10 = -m.getRow(1).x
                m20 = -m.getRow(2).x
                m01 = m.getRow(0).y
                m11 = -m.getRow(1).y
                m21 = -m.getRow(2).y
                m02 = m.getRow(0).z
                m12 = -m.getRow(1).z
                m22 = -m.getRow(2).z
            else:
                m00 = m11 = m22 = 1
                m10 = m20 = m01 = m21 = m02 = m12 = 0
        else:
            n = 1 / sqrt(1 - 2 * strength * v2 + strength * strength)
            m22 = (1 - strength * v2) * n
            m02 = strength * v0 * n
            m20 = -m02
            m12 = strength * v1 * n
            m21 = -m12

            q = 1 / q
            m00 = (v12 + v02 * m22) * q
            m11 = (v02 + v12 * m22) * q
            m01 = m10 = v0 * v1 * (m22 - 1) * q

        vec1 = Vector4(m00, m10, m20, 0)
        vec2 = Vector4(m01, m11, m21, 0)
        vec3 = Vector4(m02, m12, m22, 0)
        vec4 = Vector4(0, 0, 0, 1)
        local_m = pgl.Matrix4(vec1, vec2, vec3, vec4)
        return local_m


def grotation(m, strength):
    frame = Frame(m)
    return frame.grotation(strength)

def invTransformVector(t, v):
    x = v[0]
    y = v[1]

    v= Vector3()
    m00 = t.getRow(0).x
    m01 = t.getRow(0).y
    m02 = t.getRow(0).z
    m10 = t.getRow(1).x
    m11 = t.getRow(1).y
    m12 = t.getRow(1).z
    m20 = t.getRow(2).x
    m21 = t.getRow(2).y
    m22 = t.getRow(2).z
    d0 = m11 * m22 - m12 * m21
    d1 = m12 * m20 - m10 * m22
    d2 = m10 * m21 - m11 * m20
    d = 1.0 / (m00 * d0 + m01 * d1 + m02 * d2)
    v[0] = d0 * d * x + (m21 * m02 - m01 * m22) * d * y + (m01 * m12 - m02 * m11) * d * v[2]
    v[1] = d1 * d * x + (m00 * m22 - m02 * m20) * d * y + (m10 * m02 - m00 * m12) * d * v[2]
    v[2] = d2 * d * x + (m20 * m01 - m00 * m21) * d * y + (m00 * m11 - m01 * m10) * d * v[2]
    return v

def setFromAxisAngle(x, y, z, angle):
    n = sqrt(x * x + y * y + z * z)
    n = 1 / n
    x *= n
    y *= n
    z *= n
    c = cos(angle)
    s = sin(angle)
    omc = 1.0 - c
    m00 = c + x * x * omc
    m11 = c + y * y * omc
    m22 = c + z * z * omc
    tmp1 = x * y * omc
    tmp2 = z * s
    m01 = tmp1 - tmp2
    m10 = tmp1 + tmp2
    tmp1 = x * z * omc
    tmp2 = y * s
    m02 = tmp1 + tmp2
    m20 = tmp1 - tmp2
    tmp1 = y * z * omc
    tmp2 = x * s
    m12 = tmp1 - tmp2
    m21 = tmp1 + tmp2
    m03 = m13 = m23 = m30 = m31 = m32 = 0
    m33 = 1
    vec1 = Vector4(m00, m10, m20, m30)
    vec2 = Vector4(m01, m11, m21, m31)
    vec3 = Vector4(m02, m12, m22, m32)
    vec4 = Vector4(m03, m13, m23, m33)
    return pgl.Matrix4(vec1, vec2, vec3, vec4)

def directionalTropism(m, direction, strength):
    t = m.getColumn(2)
    x = direction[2] * t.y - direction[1] * t.z
    y = direction[0] * t.z - direction[2] * t.x
    z = direction[1] * t.x - direction[0] * t.y
    vec3 = Vector3(x, y, z)
    angle = strength * sqrt((x * x + y * y + z * z) / (t.x * t.x + t.y * t.y + t.z * t.z))
    if (angle * angle) >= 1e-20:
        vec3 = invTransformVector(m, vec3)
        return setFromAxisAngle(vec3.x, vec3.y, vec3.z, angle)
    else:
        vec1 = Vector4(1, 0, 0, 0)
        vec2 = Vector4(0, 1, 0, 0)
        vec3 = Vector4(0, 0, 1, 0)
        vec4 = Vector4(0, 0, 0, 1)
        return pgl.Matrix4(vec1, vec2, vec3, vec4)

def orthogonalTropism(m, direction, strength):
    t = m.getColumn(2)
    x = direction[2] * t.y - direction[1] * t.z
    y = direction[0] * t.z - direction[2] * t.x
    z = direction[1] * t.x - direction[0] * t.y
    vec3 = Vector3(x, y, z)
    angle = -strength * (t.x * direction[0] + t.y * direction[1] + t.z *
                         direction[2]) / sqrt(t.x * t.x + t.y * t.y + t.z * t.z)

    if (angle * angle) >= 1e-20:
        invTransformVector(m, vec3)
        return setFromAxisAngle(vec3.x, vec3.y, vec3.z, angle)
    else:
        vec1 = Vector4(1, 0, 0, 0)
        vec2 = Vector4(0, 1, 0, 0)
        vec3 = Vector4(0, 0, 1, 0)
        vec4 = Vector4(0, 0, 0, 1)
        return pgl.Matrix4(vec1, vec2, vec3, vec4)

def positionalTropism(m, target, strength):
    x = target[0] - m.getRow(0).w
    y = target[1] - m.getRow(1).w
    z = target[2] - m.getRow(2).w
    l = x * x + y * y + z * z
    t = m.getColumn(2)
    if l > 0:
        xv = z * t.y - y * t.z
        yv = x * t.z - z * t.x
        zv = y * t.x - x * t.y
        vec3 = Vector3(xv, yv, zv)
        angle = strength * sqrt((xv * xv + yv * yv + zv * zv) / (l * (t.x * t.x + t.y * t.y + t.z * t.z)))
        if (angle * angle) >= 1e-20:
            invTransformVector(m, vec3)
            return setFromAxisAngle(vec3.x, vec3.y, vec3.z, angle)
    else:
        vec1 = Vector4(1, 0, 0, 0)
        vec2 = Vector4(0, 1, 0, 0)
        vec3 = Vector4(0, 0, 1, 0)
        vec4 = Vector4(0, 0, 0, 1)
        return pgl.Matrix4(vec1, vec2, vec3, vec4)

def adjust_lu(m):
    t = m.getColumn(3)
    t = Vector3(t.x, t.y, t.z)

    m3 = pgl.Matrix3(m)
    x, y, z = Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1)
    X, Y, Z = m * Vector4(1, 0, 0, 0), m * Vector4(0, 1, 0, 0), m * Vector4(0, 0, 1, 0)
    Z = Vector3(Z.x, Z.y, Z.z)
    new_x = z ^ Z
    local_m = m
    if pgl.normSquared(new_x) > 1e-3:
        new_y = Z ^ new_x
        new_x.normalize()
        new_y.normalize()
        local_m = pgl.BaseOrientation(new_x, new_y).getMatrix()
        local_m = local_m.translation(t) * local_m
    return local_m

##########################################################################
# Debug utility

def frame(matrix, scene, color=1):
    """ Add a frame to the scene.
    The frame is represented by the matrix.
    :param color: allow to distinguish between to identical frames.
    """
    if color == 1:
        r = pgl.Material(pgl.Color3(*(255, 0, 0)))
        g = pgl.Material(pgl.Color3(*(0, 255, 0)))
        b = pgl.Material(pgl.Color3(*(0, 0, 255)))
    else:
        r = pgl.Material(pgl.Color3(*(255, 0, 255)))
        g = pgl.Material(pgl.Color3(*(255, 255, 0)))
        b = pgl.Material(pgl.Color3(*(0, 0, 0)))

    cylinder = pgl.Cylinder(radius=0.005, height=1)
    #x = pgl.AxisRotated(Vector3(0,1,0), radians(90), cylinder)
    #y = pgl.AxisRotated(Vector3(1,0,0), radians(-90), cylinder)
    z = cylinder

    #geom_x = transform4(matrix, x)
    #scene.add(pgl.Shape(geom_x, r))
    #geom_y = transform4(matrix, y)
    #scene.add(pgl.Shape(geom_y, g))
    geom_z = transform4(matrix, z)
    scene.add(pgl.Shape(geom_z, b))

##########################################################################
