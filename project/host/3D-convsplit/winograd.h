
static void winograd_f6k3_kernel_transform(
	DTYPE g[3],
	DTYPE transform[8],
	bool rescale_coefficients)
{
	const DTYPE g0 = g[0], g1 = g[1], g2 = g[2];
	/*
	 * w0 = g0
	 * w1 = ((g0 + g2) + g1) * (-2.0 / 9)
	 * w2 = ((g0 + g2) - g1) * (-2.0 / 9)
	 * w3 = ((g0 + 4 * g2) + 2 * g1) * (1.0 / 90)
	 * w4 = ((g0 + 4 * g2) - 2 * g1) * (1.0 / 90)
	 * w5 = ((g2 + 4 * g0) + 2 * g1) * (1.0 / 180)
	 * w6 = ((g2 + 4 * g0) - 2 * g1) * (1.0 / 180)
	 * w7 = g2
	 */

	/*
	 * Compute
	 *   w2 := g0 + g2
	 *   w4 := g0 + 4 * g2
	 *   w6 := g2 + 4 * g0
	 */
	const float const_4 = 4.0f;
	DTYPE w2 = g0 + g2;
	DTYPE w4 = g0 + const_4 * g2;
	DTYPE w6 = g2 + const_4 * g0;

	/*
	 * Compute
	 *   w1 = (g0 + g2) + g1
	 *   w2 = (g0 + g2) - g1
	 *   w3 = (g0 + 4 * g2) + 2 * g1
	 *   w4 = (g0 + 4 * g2) - 2 * g1
	 *   w5 = (g2 + 4 * g0) + 2 * g1
	 *   w6 = (g2 + 4 * g0) - 2 * g1
	 */
	const DTYPE two_g1 = g1 * 2.0f;
	DTYPE w1 = w2 + g1;
	w2 = w2 - g1;
	DTYPE w3 = w4 + two_g1;
	w4 = w4 - two_g1;
	DTYPE w5 = w6 + two_g1;
	w6 = w6 - two_g1;

	if (rescale_coefficients) {
		const float minus_2_over_9 = -(2.0f/9);
		w1 *= minus_2_over_9;
		w2 *= minus_2_over_9;

		const float rcp_90 = 1.0f/90;
		w3 *= rcp_90;
		w4 *= rcp_90;

		const float rcp_180 = 1.0f/180;
		w5 *= rcp_180;
		w6 *= rcp_180;
	}

	transform[0] = g0;
	transform[1] = w1;
	transform[2] = w2;
	transform[3] = w3;
	transform[4] = w4;
	transform[5] = w5;
	transform[6] = w6;
	transform[7] = g2;
}

