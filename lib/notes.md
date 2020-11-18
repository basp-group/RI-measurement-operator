# Changes, various fixes and remarks

## utils

1. `op_nufft*`, `op_p_nufft*`: two versions for the moment, depending whether the user wants to use functionalities from `irt` lirbary or not
2. `op_p_sp_wlt_basis`, `op_sp_wlt_basis`: comment the `dwtmode` instruction? (i.e., to leave the user set the boundary conditions out of the function) -> see if this possibly creates an issue somewhere in SARA or hyperSARA
3. `so_fft2_adj`: apply `real` to the final resul to avoid complex output (idealm would be `rfft2`, not available in MATLAB)
4. need to replace `power_method_op` to `op_norm` in hyperSARA and faceted HyperSARA
5. use of `op_norm_wave_par`? (from SARA-adaptive-ppd)
6. some utility functions left in hypersara -> faceted HyperSARA (check if those can be progressively removed)
