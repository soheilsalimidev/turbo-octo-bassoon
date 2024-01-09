use std::{collections::HashMap, fmt::Debug, fs::File, rc::Rc, usize, vec};

use anyhow::Result;
use nalgebra::{Dyn, OMatrix};
const RATING: &[u8] = include_bytes!("./../data/ratings.csv");
const MOIVES: &[u8] = include_bytes!("./../data/movies.csv");

#[derive(Debug, serde::Deserialize, Default, serde::Serialize)]
struct Ratings {
    user_id: usize,
    movie_id: usize,
    rating: f32,
}

#[derive(Debug, serde::Deserialize, serde::Serialize, Default)]
struct Movies {
    movie_id: usize,
    title: String,
    genres: String,
}

fn movies() -> Result<Vec<Movies>> {
    let mut rdr = csv::Reader::from_reader(MOIVES);
    Ok(rdr
        .deserialize()
        .map(|f| f.unwrap())
        .collect::<Vec<Movies>>())
}

fn reaings() -> Result<Vec<Ratings>> {
    let mut rdr = csv::Reader::from_reader(RATING);
    Ok(rdr
        .deserialize()
        .map(|f| f.unwrap())
        .collect::<Vec<Ratings>>())
}

type Matrixf = OMatrix<f32, Dyn, Dyn>;

fn main() {
    let rating = Rc::new(reaings().unwrap());
    let movies = movies().unwrap();
    // NOTE: this will result to  user_len = 610 and  moive_len = 9742
    // let (user_len , moive_len) = find_matrix_len(&rating);
    let user_id = std::env::args()
        .nth(1)
        .expect("no user given")
        .parse::<usize>()
        .unwrap();

    let mut map = HashMap::new();
    for (i, m) in movies.iter().enumerate() {
        map.insert(m.movie_id, i);
    }

    let mut rows = Vec::with_capacity(610 * 9742);
    let len = 610;
    for us in 1..(len + 1) {
        let mut row = vec![0.0; 9742];
        rating.iter().filter(|f| f.user_id == us).for_each(|f| {
            row[*map.get(&f.movie_id).unwrap()] = f.rating;
        });
        rows.append(&mut row);
    }
    let matrix = Matrixf::from_row_slice(len, 9742, &rows);
    let v_t = matrix.svd(false, true).v_t.unwrap();
    let user_row = v_t.column(user_id);
    let mut user_rating = rating.iter().filter(|f| f.user_id == user_id);
    let mut sorted_item = v_t
        .column_iter()
        .enumerate()
        .filter(|(i, _)| user_rating.find(|f| &map[&f.movie_id] == i  && f.rating != 0.0).is_none())
        .map(|(i, column)| {
            let norm = if column.norm() * user_row.norm() == 0.0 {
                1.0
            } else {
                column.norm() * user_row.norm()
            };
            (i, column.dot(&user_row) / norm)
        })
        .collect::<Vec<_>>();
    sorted_item.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    write_test_m(
        sorted_item
            .into_iter()
            .map(|(i, _)| &movies[i])
            .collect::<Vec<_>>(),
    );
    println!("result saved in output.csv")
}

fn write_test_m<T>(m: Vec<T>)
where
    T: serde::Serialize,
{
    let mut wtr = csv::Writer::from_writer(
        File::options()
            .write(true)
            .create(true)
            .truncate(true)
            .open("./output.csv")
            .unwrap(),
    );
    m.into_iter().for_each(|f| {
        wtr.serialize(f).unwrap();
    });
}

// fn find_matrix_len(reating: &Vec<Ratings>) -> (usize, usize) {
//     (
//         reating
//             .iter()
//             .map(|f| f.user_id)
//             .collect::<HashSet<u64>>()
//             .len(),
//         reating
//             .iter()
//             .map(|f| f.movie_id)
//             .collect::<HashSet<u64>>()
//             .len(),
//     )
// }
