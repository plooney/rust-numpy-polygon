extern crate numpy;
extern crate ndarray;
extern crate pyo3;
extern crate clip_rs;
extern crate rayon;

use numpy::ndarray::{Axis, s};
use numpy::{IntoPyArray, PyArray2, PyArray1};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use clip_rs::{is_point_in_path, Path, point};
use numpy::ndarray::par_azip;
use clip_rs::point::Coords;

#[pymodule]
fn polygon(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    #[pyfn(m)]
    #[pyo3(name = "points_in_polygon_new")]
    fn points_in_polygon_new<'py>(py: Python<'py>, x: &PyArray2<f64>, y: &PyArray2<f64>) -> &'py PyArray1<i8> {
        let x = x.readonly();
        let x = x.as_array();
        let y = y.readonly();
        let y = y.as_array();
        let path = y.axis_iter(Axis(0)).map(|x| point::DoublePoint::new(x[0], x[1])).collect::<Path>();

        let points = x.axis_iter(Axis(0)).map(|x| point::DoublePoint{ x:x[0], y:x[1] });
        let is_inside = points.map(|x| is_point_in_path(&x, &path)).collect::<Vec<i8>>();

        is_inside.into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "points_in_polygon_mut")]
    fn points_in_polygon_mut<'py>(_py: Python<'py>, x: &PyArray2<f64>, y: &PyArray2<f64>, res_py: &PyArray1<u8>, inds_py: &PyArray1<bool>) {
        let x = x.readonly();
        let x = x.as_array();
        let y = y.readonly();
        let y = y.as_array();
        let path = y.axis_iter(Axis(0)).map(|x| point::DoublePoint::new(x[0], x[1])).collect::<Path>();
        let mut res = unsafe { res_py.as_array_mut() };
        let inds = inds_py.readonly();
        let inds = inds.as_array();

        let points = x.axis_iter(Axis(0)).map(|x| point::DoublePoint{ x:x[0], y:x[1] }).collect::<Path>();
        par_azip!((r in &mut res, ind in &inds, &p in &points) mut_is_point_in_path(&p, &path, ind, r));
    }

    fn mut_is_point_in_path(x: &point::DoublePoint, path: &Path, ind: &bool, r: &mut u8) {
        if *ind {
            *r += is_point_in_path(x, path).abs() as u8;
            *r = *r % 2;
        }
    }

    Ok(())
}
