##Importing all the packages

from sfepy import data_dir
from sfepy.mechanics.matcoefs import stiffness_from_lame
from sfepy.discrete.fem import Mesh, FEDomain, Field
import numpy as nm
from sfepy.discrete import (FieldVariable, Material, Integral, Function,Equation, Equations, Problem)
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.base.base import IndexedStruct
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.postprocess.viewer import Viewer

##Add mesh from file
mesh = Mesh.from_file('C:/Users/16098/sfepy/meshes/2d/rectangle_tri.mesh')
domain = FEDomain('domain', mesh)
min_x, max_x = domain.get_mesh_bounding_box()[:, 0]
eps = 1e-8 * (max_x - min_x)
omega = domain.create_region('Omega', 'all')
gamma1 = domain.create_region('Gamma1','vertices in x < %.10f' % (min_x + eps),'facet')
gamma2 = domain.create_region('Gamma2','vertices in x > %.10f' % (max_x - eps),'facet')
field = Field.from_args('fu', nm.float64, 'vector', omega, approx_order=2)

u = FieldVariable('u', 'unknown', field)
v = FieldVariable('v', 'test', field, primary_var_name='u')
m = Material('m', D=stiffness_from_lame(dim=2, lam=1.0, mu=1.0))
f = Material('f', val=[[0.02], [0.01]])
integral = Integral('i', order=3)
t1 = Term.new('dw_lin_elastic(m.D, v, u)',integral, omega, m=m, v=v, u=u)
t2 = Term.new('dw_volume_lvf(f.val, v)',integral, omega, f=f, v=v)
eq = Equation('balance', t1 + t2)
eqs = Equations([eq])
fix_u = EssentialBC('fix_u', gamma1, {'u.all' : 0.0})

def shift_u_fun(ts, coors, bc=None, problem=None, shift=0.0):
    val = shift * coors[:,1]**2
    return val

bc_fun = Function('shift_u_fun', shift_u_fun,extra_args={'shift' : 0.01})
shift_u = EssentialBC('shift_u', gamma2, {'u.0' : bc_fun})

ls = ScipyDirect({})
nls_status = IndexedStruct()
nls = Newton({}, lin_solver=ls, status=nls_status)
pb = Problem('elasticity', equations=eqs)
pb.save_regions_as_groups('regions')

pb.set_bcs(ebcs=Conditions([fix_u, shift_u]))
pb.set_solver(nls)
status = IndexedStruct()
vec = pb.solve(status=status)
print('Nonlinear solver status:\n', nls_status)
print('Stationary solver status:\n', status)
pb.save_state('linear_elasticity.vtk', vec)

u = vec()
k = u.shape[0]/2
u.shape = (int(k),2)
arr = []
for i in range(0,len(u)):
    temp = u[i]
    temp2 = np.append(temp, 0.0)
    arr.append(temp2)
    
##returns vector array that can be plugged into script for clustering
np.array(arr)


