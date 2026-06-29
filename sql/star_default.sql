SELECT
	-- Basic information
	f1.object_id, f1.ra, f1.dec, f1.tract, f1.patch,
	
	-- Galactic extinction for correction
	f1.a_g, f1.a_r, f1.a_i, f1.a_z, f1.a_y,
	

	-- Number of images contributing at the object center
    -- Useful to quantify the depth of the photometry
	f1.g_inputcount_value as g_input_count,
	f1.r_inputcount_value as r_input_count,
	f1.i_inputcount_value as i_input_count,
	f1.z_inputcount_value as z_input_count,
	f1.y_inputcount_value as y_input_count,

    -- Note: it seems that S23B does not isolate the photometry based on the old R2/I2 filters and provide the magnitude correction any more. I don't know whether that's because the correction has been applied or not. Might be worth checking with the HSC team.
	-- Flag for measurements from the HSC-R2/I2 filter
	-- f1.merge_measurement_r2,
	-- f1.merge_measurement_i2,
	-- c.corr_rmag,
	-- c.corr_imag,
	
	-- CModel photometry
    -- I typically download the flux and err, but it is also possible to download the magnitude and err
    -- For my case, I like to get all the CModel fluxes, including the exponential and de Vaucouluer components as a cross-check.
	f1.g_cmodel_mag,
	f1.r_cmodel_mag,
	f1.i_cmodel_mag,
	f1.z_cmodel_mag,
	f1.y_cmodel_mag,
	f1.g_cmodel_magerr as g_cmodel_mag_err,
	f1.r_cmodel_magerr as r_cmodel_mag_err,
	f1.i_cmodel_magerr as i_cmodel_mag_err,
	f1.z_cmodel_magerr as z_cmodel_mag_err,
	f1.y_cmodel_magerr as y_cmodel_mag_err,

	-- 4. Flags for CModel photometry
	-- If the flag is set to True, it means the photometry is not reliable (final model fit failed)
    -- i-band flag is not included because we use it as the reference band, so we only include objects with successful i-band fits.
	f1.g_cmodel_flag,
	f1.r_cmodel_flag,
	f1.z_cmodel_flag,
	f1.y_cmodel_flag,
	
	-- PSF photometry
	f2.g_psfflux_mag as g_psf_mag,
	f2.r_psfflux_mag as r_psf_mag,
	f2.i_psfflux_mag as i_psf_mag,
	f2.z_psfflux_mag as z_psf_mag,
	f2.y_psfflux_mag as y_psf_mag,
        f2.g_psfflux_magerr as g_psf_mag_err,
	f2.r_psfflux_magerr as r_psf_mag_err,
	f2.i_psfflux_magerr as i_psf_mag_err,
	f2.z_psfflux_magerr as z_psf_mag_err,
	f2.y_psfflux_magerr as y_psf_mag_err,
 
	-- PSF photometry flag (for quality cut later)
	f2.g_psfflux_flag as g_psf_flag,
	f2.r_psfflux_flag as r_psf_flag,
	f2.i_psfflux_flag as i_psf_flag,
	f2.z_psfflux_flag as z_psf_flag,
	f2.y_psfflux_flag as y_psf_flag,


	-- 6. y-band flags
	f2.y_sdsscentroid_flag,
	f1.y_pixelflags_edge,
        f1.y_pixelflags_saturatedcenter,
	f1.y_pixelflags_interpolatedcenter,
	f1.y_pixelflags_crcenter,
    
    --overlaps
    f1.detect_istractinner,
    f1.detect_ispatchinner,
    
-- 5. Detailed bright star masks 
        -- It is also possible to gather more detailed information about the bright star masks here.
        msk.g_mask_brightstar_halo, 
        msk.g_mask_brightstar_ghost,
        msk.g_mask_brightstar_blooming,

        msk.r_mask_brightstar_halo,
        msk.r_mask_brightstar_ghost,
        msk.r_mask_brightstar_blooming,

        msk.i_mask_brightstar_halo,
        msk.i_mask_brightstar_ghost,
        msk.i_mask_brightstar_blooming,

        msk.z_mask_brightstar_halo,
        msk.z_mask_brightstar_ghost,
        msk.z_mask_brightstar_blooming,

        msk.y_mask_brightstar_halo,
        msk.y_mask_brightstar_ghost,
        msk.y_mask_brightstar_blooming

FROM
	s23b_wide.forced as f1
	LEFT JOIN s23b_wide.forced2 as f2 USING (object_id)
    LEFT JOIN s23b_wide.masks as msk USING (object_id)


WHERE
f1.isprimary AND
f1.tract  IN  ({$tract})                  AND
f1.i_cmodel_mag - f2.i_psfflux_mag > -0.08 AND
    
f1.i_cmodel_mag <22.0

AND NOT f1.i_cmodel_flag
AND NOT f1.g_cmodel_flag
AND NOT f1.r_cmodel_flag
AND NOT f1.z_cmodel_flag


-- Full-depth full-color cut
AND f1.g_inputcount_value >= 4
AND f1.r_inputcount_value >= 4
AND f1.i_inputcount_value >= 5
AND f1.z_inputcount_value >= 5

 -- Not failed at finding the center
AND NOT f2.g_sdsscentroid_flag
AND NOT f2.r_sdsscentroid_flag
AND NOT f2.i_sdsscentroid_flag
AND NOT f2.z_sdsscentroid_flag

-- The object's center is not outside the image
AND NOT f1.g_pixelflags_edge    
AND NOT f1.r_pixelflags_edge
AND NOT f1.i_pixelflags_edge
AND NOT f1.z_pixelflags_edge

-- Not saturated at the center  
AND NOT f1.g_pixelflags_saturatedcenter
AND NOT f1.r_pixelflags_saturatedcenter
AND NOT f1.i_pixelflags_saturatedcenter
AND NOT f1.z_pixelflags_saturatedcenter

-- The center is not interpolated
AND NOT f1.g_pixelflags_interpolatedcenter
AND NOT f1.r_pixelflags_interpolatedcenter
AND NOT f1.i_pixelflags_interpolatedcenter
AND NOT f1.z_pixelflags_interpolatedcenter

-- The center is not affected by a cosmic ray
AND NOT f1.g_pixelflags_crcenter
AND NOT f1.r_pixelflags_crcenter
AND NOT f1.i_pixelflags_crcenter
AND NOT f1.z_pixelflags_crcenter
--- NOT meas.i_pixelflags_clipped           AND --no need
--- NOT meas2.i_hsmshaperegauss_flag        AND --no need
--- meas2.i_hsmshaperegauss_sigma != 'NaN'  AND --no need
--- select galaxies: extendedness != 0
--- stars: extendedness == 0
---meas.i_extendedness_value != 0
--meas.i_cmodel_mag - meas2.i_psfflux_mag <-0.15
--meas3.i_kronflux_psf_radius - meas3.i_kronflux_radius < -0.015
;