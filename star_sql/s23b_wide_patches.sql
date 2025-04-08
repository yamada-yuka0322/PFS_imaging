SELECT

pq.skymap_id,

pq.ra,
pq.dec,

pq.tract,
pq.patch,

pq.gmag_psf_depth as g_depth,
pq.rmag_psf_depth as r_depth,
pq.imag_psf_depth as i_depth,
pq.zmag_psf_depth as z_depth,
pq.ymag_psf_depth as y_depth,

pq.gseeing,
pq.rseeing,
pq.iseeing,
pq.zseeing,
pq.yseeing

FROM
s23b_wide.patch_qa as pq

WHERE
pq.tract  IN  ({$tract})