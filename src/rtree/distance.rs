//! Distance metrics for spatial queries.
//!
//! This module provides different distance calculation methods for spatial queries,
//! including Euclidean, Haversine, and Spheroid distance calculations.

use crate::r#type::IndexableNum;
use std::f64::consts::PI;

/// A trait for calculating distances between two points.
pub trait DistanceMetric<N: IndexableNum> {
    /// Calculate the distance between two points (x1, y1) and (x2, y2).
    fn distance(&self, x1: N, y1: N, x2: N, y2: N) -> N;

    /// Calculate the distance from a point to a bounding box.
    /// This is used for spatial index optimization.
    fn distance_to_bbox(&self, x: N, y: N, min_x: N, min_y: N, max_x: N, max_y: N) -> N;

    /// Return the maximum distance value for this metric.
    fn max_distance(&self) -> N {
        N::max_value()
    }
}

/// Euclidean distance metric.
///
/// This is the standard straight-line distance calculation suitable for
/// planar coordinate systems. When working with longitude/latitude coordinates,
/// the unit of distance will be degrees.
#[derive(Debug, Clone, Copy, Default)]
pub struct EuclideanDistance;

impl<N: IndexableNum> DistanceMetric<N> for EuclideanDistance {
    #[inline]
    fn distance(&self, x1: N, y1: N, x2: N, y2: N) -> N {
        let dx = x1 - x2;
        let dy = y1 - y2;
        (dx * dx + dy * dy).sqrt().unwrap_or(N::max_value())
    }

    #[inline]
    fn distance_to_bbox(&self, x: N, y: N, min_x: N, min_y: N, max_x: N, max_y: N) -> N {
        let dx = axis_dist(x, min_x, max_x);
        let dy = axis_dist(y, min_y, max_y);
        (dx * dx + dy * dy).sqrt().unwrap_or(N::max_value())
    }
}

/// Haversine distance metric.
///
/// This calculates the great-circle distance between two points on a sphere.
/// It's more accurate for geographic distances than Euclidean distance.
/// The input coordinates should be in longitude/latitude (degrees), and
/// the output distance is in meters.
#[derive(Debug, Clone, Copy)]
pub struct HaversineDistance {
    /// Earth's radius in meters
    pub earth_radius: f64,
}

impl Default for HaversineDistance {
    fn default() -> Self {
        Self {
            earth_radius: 6378137.0, // WGS84 equatorial radius in meters
        }
    }
}

impl HaversineDistance {
    /// Create a new Haversine distance metric with custom Earth radius.
    pub fn with_radius(earth_radius: f64) -> Self {
        Self { earth_radius }
    }
}

impl<N: IndexableNum> DistanceMetric<N> for HaversineDistance {
    fn distance(&self, lon1: N, lat1: N, lon2: N, lat2: N) -> N {
        let lat1_rad = lat1.to_f64().unwrap_or(0.0) * PI / 180.0;
        let lat2_rad = lat2.to_f64().unwrap_or(0.0) * PI / 180.0;
        let delta_lat = (lat2.to_f64().unwrap_or(0.0) - lat1.to_f64().unwrap_or(0.0)) * PI / 180.0;
        let delta_lon = (lon2.to_f64().unwrap_or(0.0) - lon1.to_f64().unwrap_or(0.0)) * PI / 180.0;

        let a = (delta_lat / 2.0).sin().powi(2)
            + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

        N::from_f64(self.earth_radius * c).unwrap_or(N::max_value())
    }

    fn distance_to_bbox(
        &self,
        lon: N,
        lat: N,
        min_lon: N,
        min_lat: N,
        max_lon: N,
        max_lat: N,
    ) -> N {
        // For bbox distance with Haversine, we approximate using the closest point on the bbox
        let closest_lon = if lon < min_lon {
            min_lon
        } else if lon > max_lon {
            max_lon
        } else {
            lon
        };

        let closest_lat = if lat < min_lat {
            min_lat
        } else if lat > max_lat {
            max_lat
        } else {
            lat
        };

        self.distance(lon, lat, closest_lon, closest_lat)
    }
}

/// Spheroid distance metric.
///
/// This calculates the shortest distance between two points on the surface
/// of a spheroid (ellipsoid), providing a more accurate Earth model than
/// a simple sphere. The input coordinates should be in longitude/latitude
/// (degrees), and the output distance is in meters.
#[derive(Debug, Clone, Copy)]
pub struct SpheroidDistance {
    /// Semi-major axis (equatorial radius) in meters
    pub semi_major_axis: f64,
    /// Semi-minor axis (polar radius) in meters
    pub semi_minor_axis: f64,
}

impl Default for SpheroidDistance {
    fn default() -> Self {
        Self {
            semi_major_axis: 6378137.0,      // WGS84 equatorial radius
            semi_minor_axis: 6356752.314245, // WGS84 polar radius
        }
    }
}

impl SpheroidDistance {
    /// Create a new Spheroid distance metric with custom ellipsoid parameters.
    pub fn with_ellipsoid(semi_major_axis: f64, semi_minor_axis: f64) -> Self {
        Self {
            semi_major_axis,
            semi_minor_axis,
        }
    }

    /// Create a new Spheroid distance metric for GRS80 ellipsoid.
    pub fn grs80() -> Self {
        Self {
            semi_major_axis: 6378137.0,
            semi_minor_axis: 6356752.314140,
        }
    }
}

impl<N: IndexableNum> DistanceMetric<N> for SpheroidDistance {
    fn distance(&self, lon1: N, lat1: N, lon2: N, lat2: N) -> N {
        // Vincenty's formulae for distance on ellipsoid
        let lat1 = match lat1.to_f64() {
            Some(value) => value * PI / 180.0,
            None => return N::zero(), // Return a default value if conversion fails
        };
        let lat2 = match lat2.to_f64() {
            Some(value) => value * PI / 180.0,
            None => return N::zero(),
        };
        let delta_lon = match (lon2.to_f64(), lon1.to_f64()) {
            (Some(lon2_value), Some(lon1_value)) => (lon2_value - lon1_value) * PI / 180.0,
            _ => return N::zero(),
        };

        let a = self.semi_major_axis;
        let b = self.semi_minor_axis;
        let f = (a - b) / a; // flattening

        let u1 = ((1.0 - f) * lat1.tan()).atan();
        let u2 = ((1.0 - f) * lat2.tan()).atan();

        let sin_u1 = u1.sin();
        let cos_u1 = u1.cos();
        let sin_u2 = u2.sin();
        let cos_u2 = u2.cos();

        let mut lambda = delta_lon;
        let mut lambda_prev;
        let mut iter_limit = 100;

        let (sin_sigma, cos_sigma, sigma, _sin_alpha, cos_sq_alpha, cos_2sigma_m) = loop {
            let sin_lambda = lambda.sin();
            let cos_lambda = lambda.cos();

            let sin_sigma = ((cos_u2 * sin_lambda).powi(2)
                + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda).powi(2))
            .sqrt();

            if sin_sigma == 0.0 {
                // Co-incident points
                return N::zero();
            }

            let cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lambda;
            let sigma = sin_sigma.atan2(cos_sigma);

            let sin_alpha = cos_u1 * cos_u2 * sin_lambda / sin_sigma;
            let cos_sq_alpha = 1.0 - sin_alpha * sin_alpha;

            let cos_2sigma_m = if cos_sq_alpha == 0.0 {
                0.0 // Equatorial line
            } else {
                cos_sigma - 2.0 * sin_u1 * sin_u2 / cos_sq_alpha
            };

            let c = f / 16.0 * cos_sq_alpha * (4.0 + f * (4.0 - 3.0 * cos_sq_alpha));

            lambda_prev = lambda;
            lambda = delta_lon
                + (1.0 - c)
                    * f
                    * sin_alpha
                    * (sigma
                        + c * sin_sigma
                            * (cos_2sigma_m
                                + c * cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)));

            iter_limit -= 1;
            if iter_limit == 0 || (lambda - lambda_prev).abs() < 1e-12 {
                break (
                    sin_sigma,
                    cos_sigma,
                    sigma,
                    sin_alpha,
                    cos_sq_alpha,
                    cos_2sigma_m,
                );
            }
        };

        let u_sq = cos_sq_alpha * (a * a - b * b) / (b * b);
        let big_a =
            1.0 + u_sq / 16384.0 * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)));
        let big_b = u_sq / 1024.0 * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)));

        let delta_sigma = big_b
            * sin_sigma
            * (cos_2sigma_m
                + big_b / 4.0
                    * (cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)
                        - big_b / 6.0
                            * cos_2sigma_m
                            * (-3.0 + 4.0 * sin_sigma * sin_sigma)
                            * (-3.0 + 4.0 * cos_2sigma_m * cos_2sigma_m)));

        let distance = b * big_a * (sigma - delta_sigma);

        N::from_f64(distance).unwrap_or(N::max_value())
    }

    fn distance_to_bbox(
        &self,
        lon: N,
        lat: N,
        min_lon: N,
        min_lat: N,
        max_lon: N,
        max_lat: N,
    ) -> N {
        // For bbox distance with Spheroid, we approximate using the closest point on the bbox
        let closest_lon = if lon < min_lon {
            min_lon
        } else if lon > max_lon {
            max_lon
        } else {
            lon
        };

        let closest_lat = if lat < min_lat {
            min_lat
        } else if lat > max_lat {
            max_lat
        } else {
            lat
        };

        self.distance(lon, lat, closest_lon, closest_lat)
    }
}

/// 1D distance from a value to a range.
#[inline]
fn axis_dist<N: IndexableNum>(k: N, min: N, max: N) -> N {
    if k < min {
        min - k
    } else if k <= max {
        N::zero()
    } else {
        k - max
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_distance() {
        let metric = EuclideanDistance;
        let distance = metric.distance(0.0f64, 0.0f64, 3.0f64, 4.0f64);
        assert!((distance - 5.0f64).abs() < 1e-10);
    }

    #[test]
    fn test_haversine_distance() {
        let metric = HaversineDistance::default();
        // Distance between New York and London (approximately)
        let distance = metric.distance(-74.0f64, 40.7f64, -0.1f64, 51.5f64);
        // Should be approximately 5585 km
        assert!((distance - 5585000.0f64).abs() < 50000.0f64);
    }

    #[test]
    fn test_spheroid_distance() {
        let metric = SpheroidDistance::default();
        // Distance between New York and London (approximately)
        let distance = metric.distance(-74.0f64, 40.7f64, -0.1f64, 51.5f64);
        // Should be approximately 5585 km (slightly different from Haversine)
        assert!((distance - 5585000.0f64).abs() < 50000.0f64);
    }
}
