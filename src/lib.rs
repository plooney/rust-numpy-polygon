extern crate numpy;
extern crate pyo3;
extern crate clipper;

use numpy::ndarray::{Array1, ArrayView1, ArrayView2, Axis, s};
use numpy::{IntoPyArray, PyArray2, PyArray1};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use clipper::{is_point_in_path, Path, point};

#[pymodule]
fn polygon(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    fn point_in_polygon(pt: ArrayView1<'_, i32>, y: ArrayView2<'_, i32>) -> i8 {
        //returns 0 if false, +1 if true, -1 if pt ON polygon boundary
        /*
          int result = 0;
          size_t cnt = path.size();
          if (cnt < 3) return 0;
          IntPoint ip = path[0];
          for(size_t i = 1; i <= cnt; ++i)
          {
            IntPoint ipNext = (i == cnt ? path[0] : path[i]);
            if (ipNext.Y == pt.Y)                          
            {                                              
                if ((ipNext.X == pt.X) || (ip.Y == pt.Y && 
                  ((ipNext.X > pt.X) == (ip.X < pt.X)))) return -1;
            }
            if ((ip.Y < pt.Y) != (ipNext.Y < pt.Y))
            {
              if (ip.X >= pt.X)
              {
                if (ipNext.X > pt.X) result = 1 - result;
                else                                                     
                {
                  double d = (double)(ip.X - pt.X) * (ipNext.Y - pt.Y) -
                    (double)(ipNext.X - pt.X) * (ip.Y - pt.Y);
                  if (!d) return -1;
                  if ((d > 0) == (ipNext.Y > ip.Y)) result = 1 - result;
                }   
              } else
              {
                if (ipNext.X > pt.X)                                     
                {
                  double d = (double)(ip.X - pt.X) * (ipNext.Y - pt.Y) -
                    (double)(ipNext.X - pt.X) * (ip.Y - pt.Y);
                  if (!d) return -1;
                  if ((d > 0) == (ipNext.Y > ip.Y)) result = 1 - result;
                }
              }
            }
            ip = ipNext;
          } 
          return result;
        */
        let mut result = 0;
        let num_points = y.shape()[0];
        if num_points < 3 { return 0 }
        //let y = arr2(&[[ 6,  5,  4],
        //       [12, 11, 10]]);
        let mut ip = y.slice(s![0, ..]);
        
        for i in 1..num_points+1 {
            let ip_next = if i == num_points {y.slice(s![0, ..])} else {y.slice(s![i, ..])}; 
            let ip_x = ip[0 as usize];
            let ip_y = ip[1 as usize];
            if ip_next[1] == pt[1] {
                if (ip_next[0] == pt[0]) || (ip_next[1] == pt[1] && ((ip_next[0] > pt[0]) == (ip_x < pt[0]))) {
                    return -1
                }
            }
            if (ip_next[1] < pt[1]) != (ip_y < pt[1]) {
                if ip_x >= pt[0] {
                    if ip_next[0] > pt[0] {result = 1 - result}
                    else {
                        let d = (ip_x-pt[0])*(ip_next[1]-pt[1]) - (ip_next[0]-pt[0])*(ip_y-pt[1]);
                        if d==0 {return -1}
                        if (d > 0) == (ip_next[1] > ip_y) {result = 1 - result}
                    }
                } else {
                    if ip_next[0] > pt[0] {
                        let d = (ip_x-pt[0])*(ip_next[1]-pt[1]) - (ip_next[0]-pt[0])*(ip_y-pt[1]);
                        if d==0 {return -1}
                        if (d > 0) == (ip_next[1] > ip_y) {result = 1 - result}
                    }
                }
            }
            ip = ip_next;
        }
        result 
    }

    #[pyfn(m)]
    #[pyo3(name = "points_in_polygon")]
    fn points_in_polygon<'py>(py: Python<'py>, x: &PyArray2<i32>, y: &PyArray2<i32>) -> &'py PyArray1<i8> {
        let x = x.readonly();
        let x = x.as_array();
        let y = y.readonly();
        let y = y.as_array();

        let no_of_points = x.shape()[0];
        let mut is_inside = Array1::<i8>::zeros(no_of_points);
        let iter = x.axis_iter(Axis(0)).zip(is_inside.iter_mut());

        for (p, b) in iter {
            *b = point_in_polygon(p, y);
        }
        is_inside.into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "points_in_polygon_new")]
    fn points_in_polygon_new<'py>(py: Python<'py>, x: &PyArray2<i64>, y: &PyArray2<i64>) -> &'py PyArray1<i8> {
        let x = x.readonly();
        let x = x.as_array();
        let y = y.readonly();
        let y = y.as_array();
        let vec = y.axis_iter(Axis(0)).map(|x| point::IntPoint::new(x[0], x[1])).collect::<Vec<point::IntPoint2d>>();

        let path = Path{poly: vec};
        let points = x.axis_iter(Axis(0)).map(|x| point::IntPoint2d{ x:x[0], y:x[1] });
        let is_inside = points.map(|x| is_point_in_path(&x, &path)).collect::<Vec<i8>>();

        is_inside.into_pyarray(py)
    }

    Ok(())
}
