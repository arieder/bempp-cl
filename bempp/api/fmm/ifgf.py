"""Main interface class to IFGF."""
import numpy as _np
import atexit as _atexit
#from juliacall import Main as jl
#jl.include("/home/arieder/devel/bempp-cl-mod/bempp/api/fmm/bempp_ifgf.jl");

from pathlib import Path
import atexit
#from julia import Main
file_dir = Path(__file__).parent
#Main.include(f"{file_dir}/bempp_ifgf.jl")

from multiprocessing import shared_memory
import subprocess
import signal
import ctypes
import os
    
FMM_TMP_DIR = None

import pyifgf
       


class IFGFInterface(object):
    """Interface to IFGF."""

    def __init__(
        self,
        source_points,
        target_points,
        source_normals,
        target_normals,
        mode,
        wavenumber=None,
        precision="double",
        singular_correction=None,
        is_dl=False
    ):

        """Instantiate an IFGF session."""
        import bempp.api
        import os
        from bempp.api.utils.helpers import create_unique_id

        self._singular_correction = singular_correction

        self._mode = mode

        self.is_ifgf=True

        import bempp.api
        order=bempp.api.GLOBAL_PARAMETERS.ifgf.order
        tol=bempp.api.GLOBAL_PARAMETERS.ifgf.tol
        leaf_size=bempp.api.GLOBAL_PARAMETERS.ifgf.leaf_size
        n_elements=bempp.api.GLOBAL_PARAMETERS.ifgf.n_elements
        

        if(is_dl==True):
            self.op=pyifgf.DoubleLayerHelmholtzIfgfOperator(-1j*wavenumber,leaf_size,order,n_elements,tol)
            self.op.init(source_points.T,target_points.T,source_normals.T);

        else:
            self.op=pyifgf.GradHelmholtzIfgfOperator(-1j*wavenumber,leaf_size,order,n_elements,tol)
            self.op.init(source_points.T,target_points.T);

        

        self.is_dl=is_dl
        self.wavenumber=wavenumber
        if mode == "laplace":
            self._kernel_parameters = _np.array([], dtype="float64")
        elif mode == "helmholtz":
            self._kernel_parameters = _np.array(
                [_np.real(wavenumber), _np.imag(wavenumber)], dtype="float64"
            )
        elif mode == "modified_helmholtz":
            self._kernel_parameters = _np.array([wavenumber], dtype="float64")
        



    @property
    def number_of_source_points(self):
        """Return number of source points."""
        return len(self._source_points)

    @property
    def number_of_target_points(self):
        """Return number of target points."""
        return len(self._target_points)

    def evaluate(self, vec, apply_singular_correction=True,deriv=0,wavenumber=-1,source_normals=None):
        """Evalute the Fmm."""
        import bempp.api
        from bempp.api.fmm.helpers import debug_fmm

        with bempp.api.Timer(message="Evaluating ifgf."):

            with bempp.api.Timer(message="Calling IFGF."):
                if bempp.api.GLOBAL_PARAMETERS.fmm.dense_evaluation:
                    from bempp.api.fmm.helpers import dense_interaction_evaluator

                    result = dense_interaction_evaluator(
                        self._target_points,
                        self._source_points,
                        vec,
                        self._mode,
                        self._kernel_parameters,
                    )
                else:
                    if(not self.is_dl):
                        self.op.setDx(deriv-1);

                        
                    r1_c=self.op.mult(vec);
                    result = r1_c

                    
                if bempp.api.GLOBAL_PARAMETERS.fmm.debug:
                    debug_fmm(
                        self._target_points,
                        self._source_points,
                        vec,
                        self._mode,
                        self._kernel_parameters,
                        result,
                    )
            print("starting singular correction")
            if apply_singular_correction and self._singular_correction is not None:
                if(self.is_dl):
                    result -= (self._singular_correction @ (source_normals[:,0]*vec)).reshape([-1, 4])[:,1]
                    result -= (self._singular_correction @ (source_normals[:,1]*vec)).reshape([-1, 4])[:,2]
                    result -= (self._singular_correction @ (source_normals[:,2]*vec)).reshape([-1, 4])[:,3]
                    
                else:
                    result -= (self._singular_correction @ vec).reshape([-1, 4])[:,deriv]

            print("done with singular corrections")

            return result

    def as_matrix(self):
        """Return matrix representation of Fmm."""
        import numpy as np

        ident = np.identity(self.number_of_source_points)

        res = np.zeros(
            (self.number_of_target_points, self.number_of_source_points),
            dtype="float64",
        )

        for index in range(self.number_of_source_points):
            res[:, index] = self.evaluate(ident[:, index])[:, 0]

        return res

    @classmethod
    def from_grid(
        cls,
        source_grid,
        mode,
        wavenumber=None,
        target_grid=None,
        precision="double",
        device_interface=None,
        is_dl=False,
        domain=None,
        dual_to_range=None
    ):
        """
        Initialise an IFGF instance from a given source and target grid.

        Parameters
        ----------
        source_grid : Grid object
            Grid for the source points.
        mode: string
            Fmm mode. One of 'laplace', 'helmholtz', or 'modified_helmholtz'
        wavenumber : real number
            For Helmholtz or modified Helmholtz the wavenumber.
        target_grid : Grid object
            An optional target grid. If not provided the source and target
            grid are assumed to be identical.
        precision : string
            Either 'single' or 'double'. Currently, the Fmm is always
            executed in double precision.
        device_interface : string
            Either 'numba' or 'opencl'. If not provided, the DEFAULT_DEVICE_INTERFACE
            will be used for the calculation of local interactions.
        """
        import bempp.api
        from bempp.api.integration.triangle_gauss import rule
        from bempp.api.fmm.helpers import get_local_interaction_operator
        from bempp.api.integration.triangle_gauss import get_number_of_quad_points
        import numpy as np

        quadrature_order = bempp.api.GLOBAL_PARAMETERS.quadrature.regular

        local_points, weights = rule(quadrature_order)

        if target_grid is None:
            target_grid = source_grid

        source_points = source_grid.map_to_point_cloud(
            quadrature_order, precision=precision
        )

        if target_grid != source_grid:
            target_points = target_grid.map_to_point_cloud(
                quadrature_order, precision=precision
            )
        else:
            target_points = source_points

            
        npoints = get_number_of_quad_points(quadrature_order)
        source_normals = get_normals(domain, npoints)
        target_normals = get_normals(dual_to_range, npoints)

            
        singular_correction = None

        if target_grid == source_grid:
            # Require singular correction terms.

            if mode == "laplace":
                singular_correction = get_local_interaction_operator(
                    source_grid,
                    local_points,
                    "laplace",
                    np.array([], dtype="float64"),
                    precision,
                    False,
                    device_interface,
                )
            elif mode == "helmholtz":
                singular_correction = get_local_interaction_operator(
                    source_grid,
                    local_points,
                    "helmholtz",
                    np.array(
                        [_np.real(wavenumber), _np.imag(wavenumber)], dtype="float64"
                    ),
                    precision,
                    True,
                    device_interface,
                )
            elif mode == "modified_helmholtz":
                singular_correction = get_local_interaction_operator(
                    source_grid,
                    local_points,
                    "modified_helmholtz",
                    np.array([wavenumber], dtype="float64"),
                    precision,
                    False,
                    device_interface,
                )

                

                
        return cls(
            source_points,
            target_points,
            source_normals,
            target_normals,
            mode,
            wavenumber=wavenumber,
            precision=precision,
            singular_correction=singular_correction,
            is_dl=is_dl
        )

def get_normals(space, npoints):
    """Get the normal vectors on the quadrature points."""
    import numpy as np

    grid = space.grid
    number_of_elements = grid.number_of_elements

    normals = np.empty((npoints * number_of_elements, 3), dtype="float64")
    for element in range(number_of_elements):
        for n in range(npoints):
            normals[npoints * element + n, :] = (
                grid.normals[element] * space.normal_multipliers[element]
            )
    return normals
