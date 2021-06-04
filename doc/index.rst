.. toctree::
   :maxdepth: 2
   :caption: Contents:

Numerical simulations of large-deforming fluid-saturated porous media using an Eulerian incremental formulation
===============================================================================================================
   

Running simulation
------------------

To run the numerical simulation, download the
`archive <https://github.com/sfepy/example_largedef_porodyn/archive/v1.zip>`_,
unpack it in the main SfePy (see [CimrmanLukesRohan2019]_) directory and type:

.. code::

   ./simple.py example_largedef_porodyn-1/porodynhe_example2d.py

This invokes the time-stepping solver for the problem of a large-deforming porous structure,
see [RohanLukes2017]_ for details.

The computed results can be visualized using the :code:`resview.py`
post-processing script as follows:

.. code::

   ./resview.py example_largedef_porodyn-1/results/rect_*.vtk -w u -f p:p0 -v "0,0" -s 1


References
----------

.. [RohanLukes2017] Rohan E., Lukeš V.
   Modeling large-deforming fluid-saturated porous media using an Eulerian incremental formulation.
   Advances in Engineering Software, 113:84-95, 2017,
   `DOI:10.1016/j.advengsoft.2016.11.003 <https://doi.org/10.1016/j.advengsoft.2016.11.003>`_

.. [CimrmanLukesRohan2019] Cimrman R., Lukeš V., Rohan E.
   Multiscale finite element calculations in Python using SfePy.
   Advances in Computational Mathematics, 45(4):1897-1921, 2019,
   `DOI:10.1007/s10444-019-09666-0 <https://doi.org/10.1007/s10444-019-09666-0>`_
