SELECT
forced.object_id
, forced.parent_id
, forced.tract
, forced.patch
, forced.ra
, forced.dec
,forced2.i_psfflux_mag as i_psfflux_mag
    
-- forced measurement (extinction needed, subtract this to get magnitude)
, forced.a_g
, forced.a_r
,forced.a_i
, forced.a_z
, forced.a_y

-- forced CModel magnitudes and fluxes (needed)
, forced.g_cmodel_mag       as forced_g_cmodel_mag
, forced.g_cmodel_magerr    as forced_g_cmodel_magerr

, forced.r_cmodel_mag       as forced_r_cmodel_mag
, forced.r_cmodel_magerr    as forced_r_cmodel_magerr

, forced.i_cmodel_mag       as forced_i_cmodel_mag
, forced.i_cmodel_magerr    as forced_i_cmodel_magerr

, forced.z_cmodel_mag       as forced_z_cmodel_mag
, forced.z_cmodel_magerr    as forced_z_cmodel_magerr
    
, forced.y_cmodel_mag       as forced_y_cmodel_mag
, forced.y_cmodel_magerr    as forced_y_cmodel_magerr


FROM
s21a_wide.meas as meas
LEFT JOIN s21a_wide.meas2 as meas2 using (object_id)
LEFT JOIN s21a_wide.forced as forced using (object_id)
LEFT JOIN s21a_wide.forced2 as forced2 using (object_id)
LEFT JOIN s21a_wide.masks as mask using (object_id)

WHERE
meas.tract  IN  ({$tract})                  AND
NOT meas.i_deblend_skipped                  AND
NOT meas2.i_sdsscentroid_flag               AND
NOT meas.i_pixelflags_edge                  AND
NOT meas.i_pixelflags_interpolatedcenter    AND
NOT meas.i_pixelflags_saturatedcenter       AND
NOT meas.i_pixelflags_crcenter              AND
NOT meas.i_pixelflags_bad                   AND
NOT meas.i_pixelflags_suspectcenter         AND
NOT meas.i_pixelflags_clipped               AND
meas.i_detect_isprimary                     AND
forced.i_cmodel_mag - forced2.i_psfflux_mag > -0.08 AND
    
forced.i_cmodel_mag - forced.a_i <22.0
--- NOT meas.i_pixelflags_clipped           AND --no need
--- NOT meas2.i_hsmshaperegauss_flag        AND --no need
--- meas2.i_hsmshaperegauss_sigma != 'NaN'  AND --no need
--- select galaxies: extendedness != 0
--- stars: extendedness == 0
---meas.i_extendedness_value != 0
--meas.i_cmodel_mag - meas2.i_psfflux_mag <-0.15
--meas3.i_kronflux_psf_radius - meas3.i_kronflux_radius < -0.015
ORDER BY forced.object_id