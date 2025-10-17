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
	
        -- Measured photometry for star-galaxy separation 
        -- cmodel magnitudes 
        m1.g_cmodel_mag as g_meas_cmodel_mag,
        m1.r_cmodel_mag as r_meas_cmodel_mag,
        m1.i_cmodel_mag as i_meas_cmodel_mag,
        m1.z_cmodel_mag as z_meas_cmodel_mag,
        m1.y_cmodel_mag as y_meas_cmodel_mag,
        m1.i_cmodel_flag as i_meas_cmodel_flag,
        
        -- psf magnitudes 
        m2.g_psfflux_mag as g_meas_psf_mag,
        m2.r_psfflux_mag as r_meas_psf_mag,
        m2.i_psfflux_mag as i_meas_psf_mag,
        m2.z_psfflux_mag as z_meas_psf_mag,
        m2.y_psfflux_mag as y_meas_psf_mag,
        m2.i_psfflux_flag as i_meas_psf_flag,

	    -- PSF-corrected aperture photometry

        -- Blendedness and deblender-related diagnoses.
        m2.g_deblend_blendedness as g_blendedness,
        m2.r_deblend_blendedness as r_blendedness,
        m2.i_deblend_blendedness as i_blendedness,
        m2.z_deblend_blendedness as z_blendedness,
        m2.y_deblend_blendedness as y_blendedness,
        
        -- some footprints  along very bright objects, ghosts, streaks, and
        -- other similar artifacts are not blended (see Issue #7) 
        m1.deblend_skipped,

        -- This is the "de-noised" version of the blendedness (see Bosch et al. 2018)
        m2.g_blendedness_abs,
        m2.r_blendedness_abs,
        m2.i_blendedness_abs,
        m2.z_blendedness_abs,
        m2.y_blendedness_abs,

        -- The maxoverlap metric: Maximum overlap with all of the other neighbors flux combined (not familiar with this, but sounds useful).
        m2.g_deblend_maxoverlap, 
        m2.r_deblend_maxoverlap,
        m2.i_deblend_maxoverlap,
        m2.z_deblend_maxoverlap,
        m2.y_deblend_maxoverlap,

        -- The deblender flag
        m2.g_blendedness_flag,
        m2.r_blendedness_flag,
        m2.i_blendedness_flag,
        m2.z_blendedness_flag,
        m2.y_blendedness_flag,


        -- aperture photometry used for low surface brightness object cut
        -- (Issue #10) 
        m3.g_apertureflux_10_mag,
        m3.r_apertureflux_10_mag,
        m3.i_apertureflux_10_mag,
        m3.z_apertureflux_10_mag,
        m3.y_apertureflux_10_mag,
        
        m3.g_apertureflux_10_flag,
        m3.r_apertureflux_10_flag,
        m3.i_apertureflux_10_flag,
        m3.z_apertureflux_10_flag,
        m3.y_apertureflux_10_flag,
	
	-- Flags for later selection
	-- 1. The general failure flag
	f1.g_pixelflags,
	f1.r_pixelflags,
	f1.i_pixelflags,
	f1.z_pixelflags,
	f1.y_pixelflags,
	
	-- 2. Saturated or interpolated pixels on the footprint (not center)
	f1.g_pixelflags_saturated,
	f1.r_pixelflags_saturated,
	f1.i_pixelflags_saturated,
	f1.z_pixelflags_saturated,
	f1.y_pixelflags_saturated,
	f1.g_pixelflags_interpolated,
	f1.r_pixelflags_interpolated,
	f1.i_pixelflags_interpolated,
	f1.z_pixelflags_interpolated,
	f1.y_pixelflags_interpolated,

	-- 3. Other pixel flags
	f1.g_pixelflags_bad,
	f1.r_pixelflags_bad,
	f1.i_pixelflags_bad,
	f1.z_pixelflags_bad,
	f1.y_pixelflags_bad,
	f1.g_pixelflags_suspectcenter,
	f1.r_pixelflags_suspectcenter,
	f1.i_pixelflags_suspectcenter,
	f1.z_pixelflags_suspectcenter,
	f1.y_pixelflags_suspectcenter,
	f1.g_pixelflags_clippedcenter,
	f1.r_pixelflags_clippedcenter,
	f1.i_pixelflags_clippedcenter,
	f1.z_pixelflags_clippedcenter,
	f1.y_pixelflags_clippedcenter,

	-- 4. Bright object masks
	f1.g_pixelflags_bright_object,
	f1.r_pixelflags_bright_object,
	f1.i_pixelflags_bright_object,
	f1.z_pixelflags_bright_object,
	f1.y_pixelflags_bright_object,

        -- 5. Detailed bright star masks 
        -- It is also possible to gather more detailed information about the bright star masks here.
        msk.g_mask_brightstar_halo, 
        msk.g_mask_brightstar_dip, 
        msk.g_mask_brightstar_ghost,
        msk.g_mask_brightstar_blooming,
        msk.g_mask_brightstar_ghost12,
        msk.g_mask_brightstar_ghost15,

        msk.r_mask_brightstar_halo,
        msk.r_mask_brightstar_dip,
        msk.r_mask_brightstar_ghost,
        msk.r_mask_brightstar_blooming,
        msk.r_mask_brightstar_ghost12,
        msk.r_mask_brightstar_ghost15,

        msk.i_mask_brightstar_halo,
        msk.i_mask_brightstar_dip,
        msk.i_mask_brightstar_ghost,
        msk.i_mask_brightstar_blooming,
        msk.i_mask_brightstar_ghost12,
        msk.i_mask_brightstar_ghost15,

        msk.z_mask_brightstar_halo,
        msk.z_mask_brightstar_dip,
        msk.z_mask_brightstar_ghost,
        msk.z_mask_brightstar_blooming,
        msk.z_mask_brightstar_ghost12,
        msk.z_mask_brightstar_ghost15,

        msk.y_mask_brightstar_halo,
        msk.y_mask_brightstar_dip,
        msk.y_mask_brightstar_ghost,
        msk.y_mask_brightstar_blooming,
        msk.y_mask_brightstar_ghost12,
        msk.y_mask_brightstar_ghost15,

	-- 6. y-band flags
	f2.y_sdsscentroid_flag,
	f1.y_pixelflags_edge,
        f1.y_pixelflags_saturatedcenter,
	f1.y_pixelflags_interpolatedcenter,
	f1.y_pixelflags_crcenter,

FROM
	s23b_wide.forced as f1
	LEFT JOIN s23b_wide.forced2 as f2 USING (object_id)
	LEFT JOIN s23b_wide.forced4 as f4 USING (object_id)
	LEFT JOIN s23b_wide.forced5 as f5 USING (object_id)
	LEFT JOIN s23b_wide.forced6 as f6 USING (object_id)
	LEFT JOIN s23b_wide.meas as m1 USING (object_id)
	LEFT JOIN s23b_wide.meas2 as m2 USING (object_id)
	LEFT JOIN s23b_wide.meas3 as m3 USING (object_id)
        LEFT JOIN s23b_wide.masks as msk USING (object_id)


WHERE
m1.tract  IN  ({$tract})                  AND
f1.i_cmodel_mag - f2.i_psfflux_mag > -0.08 AND
    
f1.i_cmodel_mag - f1.a_i <22.0

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