from __future__ import print_function
# gravity deposition, continuing with oedometric test after stabilization
# shows also how to run parametric studies with yade-batch

# The components of the batch are:
# 1. table with parameters, one set of parameters per line (ccc.table)
# 2. utils.readParamsFromTable which reads respective line from the parameter file
# 3. the simulation muse be run using yade-batch, not yade
#
# $ yade-batch --job-threads=1 03-oedometric-test.table 03-oedometric-test.py
#


# create box with free top, and ceate loose packing inside the box
from yade import plot, polyhedra_utils
from yade import qt
import numpy as np


def object_v(center, length, width, height, rotation):
    v = np.array([[-length/2, -width/2, -height/2],
                  [length/2, -width/2, -height/2],
                  [length/2, width/2, -height/2],
                  [-length/2, width/2, -height/2],
                  [-width/2, -length/2, height/2],
                  [width/2, -length/2, height/2],
                  [width/2, length/2, height/2],
                  [-width/2, length/2, height/2]])

    rotation_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    if rotation:
        v = np.transpose(np.matmul(rotation_matrix, np.transpose(v)))
    v = v + center

    return v


# def checkUnbalanced():   
#     # at the very start, unbalanced force can be low as there is only few contacts, but it does not mean the packing is stable
#     print("unbalanced forces = %.5f, position %f, %f, %f"%(utils.unbalancedForce(), t.state.pos[0], t.state.pos[1], t.state.pos[2]))
       

def yade_simulate():
    m = PolyhedraMat()
    m.density = 1e3 #kg/m^3 
    # m.young = 1E6 #Pa
    # m.poisson = 20000/1E6
    m.young = 1E9 #Pa
    m.poisson = 0.4
    m.frictionAngle = 0.6 #rad

    m_particle = PolyhedraMat()
    m_particle.density = 1e10 #kg/m^3 
    m_particle.young = 1E12 #Pa
    m_particle.poisson = 0.4
    m_particle.frictionAngle = 0.6 #rad

    O.bodies.append(utils.wall(0, axis=2, sense=1, material=m))

    base = 1
    offset = 0.5
    width = base - offset
    length = base + offset
    height = base
    center_distance = 1
    center_height = 5

    n_row, n_col = (5, 5)
    x = np.linspace(base/2, base/2 + (n_col - 1) * center_distance, n_col)
    y = np.linspace(base/2 + (n_row - 1) * center_distance, base/2, n_row)
    xv, yv = np.meshgrid(x, y)

    for i in range(n_row):
        for j in range(n_col):
            center = np.array([xv[i, j], yv[i, j], center_height])
            rotation = (i + j) % 2 == 0
            fixed = True if i == 0 or j == 0 or i == n_row - 1 or j == n_col - 1 else False
            v = object_v(center, length, width, height, rotation)
            t = polyhedra_utils.polyhedra(m, v=v, fixed=fixed)
            O.bodies.append(t)


    length_particle = base/10
    v_particle = np.array([[-length_particle/2, -length_particle/2, -length_particle/2],
                  [length_particle/2, -length_particle/2, -length_particle/2],
                  [length_particle/2, length_particle/2, -length_particle/2],
                  [-length_particle/2, length_particle/2, -length_particle/2],
                  [-length_particle/2, -length_particle/2, length_particle/2],
                  [length_particle/2, -length_particle/2, length_particle/2],
                  [length_particle/2, length_particle/2, length_particle/2],
                  [-length_particle/2, length_particle/2, length_particle/2]])
    center_particle = np.array([(xv[0, 0] + xv[-1, - 1])/2, (yv[0, 0] + yv[-1, -1])/2, center_height + height/2 + length_particle/2])
    v_particle += center_particle
    t = polyhedra_utils.polyhedra(m_particle, v=v_particle)
    # O.bodies.append(t)

    O.engines=[
       ForceResetter(),
       InsertionSortCollider([Bo1_Polyhedra_Aabb(), Bo1_Wall_Aabb(), Bo1_Facet_Aabb()]),
       InteractionLoop(
          [Ig2_Wall_Polyhedra_PolyhedraGeom(), Ig2_Polyhedra_Polyhedra_PolyhedraGeom(), Ig2_Facet_Polyhedra_PolyhedraGeom()], 
          [Ip2_PolyhedraMat_PolyhedraMat_PolyhedraPhys()], # collision "physics"
          [Law2_PolyhedraGeom_PolyhedraPhys_Volumetric()]   # contact law -- apply forces
       ),
       #GravityEngine(gravity=(0,0,-9.81)),
       NewtonIntegrator(damping=0.5, gravity=(0,0, -9.81)),
       # PyRunner(command='checkUnbalanced()',realPeriod=3,label='checker')
    ]


    # O.dt = 0.025*polyhedra_utils.PWaveTimeStep()
    O.dt = 0.001
 
    qt.Controller()
    V = qt.View()

    O.saveTmp()
    #O.run()
    #O.save('./done')
    utils.waitIfBatch()


if __name__ == '__main__':
    yade_simulate()

