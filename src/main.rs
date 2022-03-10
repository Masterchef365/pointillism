use anyhow::{Context, Result};
use rand::prelude::*;
use rand::SeedableRng;
use std::fs::File;
use std::io::BufWriter;
use std::ops::{Add, Sub};
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(StructOpt)]
struct Opt {
    /// Input image path, must be 8-bit RGB
    #[structopt()]
    input: PathBuf,

    /// Output image path
    #[structopt(short, long, default_value = "out.png")]
    output: PathBuf,

    /// Number of points to attempt
    #[structopt(short, long, default_value = "100000")]
    n_points: usize,

    /// Minimum point size
    #[structopt(long = "pmin", default_value = "3")]
    min_point_size: isize,

    /// Minimum point size
    #[structopt(long = "pmax", default_value = "10")]
    max_point_size: isize,

    /// Random seed (unsigned integer)
    #[structopt(short, long)]
    seed: Option<u64>,
}

fn main() -> Result<()> {
    // Parse args
    let args = Opt::from_args();

    // Read image
    let input_img = read_png(&args.input).context("Failed to read input image")?;

    // Create output image
    let mut output_img = RgbImage::new(input_img.width(), input_img.height());

    // Seed
    let seed = args.seed.unwrap_or_else(|| rand::thread_rng().gen());
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    println!("Using seed {}", seed);

    //let mut cost_ring = vec![0.0; 5000];
    //let mut cost_ring_idx = 0;

    // Draw points
    'points: for point_idx in 0..args.n_points {
        // Sample a random pixel from the input image
        let sample_pos = random_pos(&mut rng, input_img.width(), input_img.height());
        let color = input_img.get(sample_pos).unwrap();

        // Select a random radius and point position in the input image
        let r = rng.gen_range(args.min_point_size..=args.max_point_size);
        let point_pos = random_pos(&mut rng, input_img.width(), input_img.height());

        let shape = || stroke_circle(r).map(|c| point_pos + c);

        // Calculate the cost for this dot
        let mut total_cost_before = 0.0;
        let mut total_cost_after = 0.0;
        for coord in shape() {
            let input_px = match input_img.get(coord) {
                Some(px) => px,
                None => continue 'points,
            };

            let output_px = output_img.get(coord).unwrap();

            total_cost_before += cost(input_px, output_px);
            total_cost_after += cost(input_px, color);
        }

        // If the cost is an improvement, add the dot to the image
        if total_cost_after < total_cost_before {
            for coord in shape() {
                output_img.set(coord, color);
            }
        }

        //cost_ring[cost_ring_idx] = total_cost_after.min(total_cost_before);
        //cost_ring_idx = cost_ring_idx % cost_ring.len();

        if point_idx % 10_000 == 0 {
            //let avg = cost_ring.iter().copied().sum::<f32>() / cost_ring.len() as f32;
            //println!("{}/{} = {:02}%, Avg: {}", point_idx, args.n_points, point_idx as f32 * 100.0 / args.n_points as f32, avg);
            println!(
                "{}/{} = {:02}%",
                point_idx,
                args.n_points,
                point_idx as f32 * 100.0 / args.n_points as f32
            );
        }
    }

    // Write output image
    write_png(&args.output, &output_img)
}

fn random_pos(rng: &mut impl Rng, width: usize, height: usize) -> Coord {
    Coord(
        rng.gen_range(0..width) as isize,
        rng.gen_range(0..height) as isize,
    )
}

fn cost(truth: [u8; 3], sample: [u8; 3]) -> f32 {
    truth
        .into_iter()
        .zip(sample)
        .map(|(t, s)| t as f32 - s as f32)
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt()
}

fn read_png(path: &PathBuf) -> Result<RgbImage> {
    let decoder = png::Decoder::new(File::open(path)?);

    let mut reader = decoder.read_info()?;
    let mut buf = vec![0; reader.output_buffer_size()];

    let info = reader.next_frame(&mut buf)?;

    assert!(info.color_type == png::ColorType::Rgb);
    assert!(info.bit_depth == png::BitDepth::Eight);

    buf.truncate(info.buffer_size());

    Ok(RgbImage::from_array(buf, info.width as _, info.height as _))
}

fn write_png(path: &PathBuf, image: &RgbImage) -> Result<()> {
    let file = File::create(path)?;
    let ref mut w = BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, image.width as _, image.height as _); // Width is 2 pixels and height is 1.
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);

    let mut writer = encoder.write_header()?;
    writer.write_image_data(&image.data)?; // Save

    Ok(())
}

struct RgbImage {
    data: Vec<u8>,
    width: usize,
    height: usize,
}

impl RgbImage {
    pub fn new(width: usize, height: usize) -> Self {
        Self::from_array(vec![0; width * height * 3], width, height)
    }

    pub fn from_array(data: Vec<u8>, width: usize, height: usize) -> Self {
        assert_eq!(data.len(), width * height * 3);
        Self {
            data,
            width,
            height,
        }
    }

    fn calc_index(&self, Coord(x, y): Coord) -> Option<usize> {
        let b = x >= 0 && y >= 0;
        let (x, y) = (x as usize, y as usize);
        let b = b && x < self.width && y < self.height;
        b.then(|| (x + y * self.width) * 3)
    }

    pub fn get(&self, coord: Coord) -> Option<[u8; 3]> {
        self.calc_index(coord)
            .map(|idx| [self.data[idx + 0], self.data[idx + 1], self.data[idx + 2]])
    }

    pub fn set(&mut self, coord: Coord, [r, g, b]: [u8; 3]) {
        if let Some(idx) = self.calc_index(coord) {
            self.data[idx + 0] = r;
            self.data[idx + 1] = g;
            self.data[idx + 2] = b;
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Coord(pub isize, pub isize);

impl Add for Coord {
    type Output = Coord;
    fn add(self, Coord(dx, dy): Self) -> Self {
        let Coord(x, y) = self;
        Coord(x + dx, y + dy)
    }
}

impl Sub for Coord {
    type Output = Coord;
    fn sub(self, Coord(dx, dy): Self) -> Self {
        let Coord(x, y) = self;
        Coord(x - dx, y - dy)
    }
}

fn fill_circle(r: isize) -> impl Iterator<Item = Coord> {
    (-r..=r)
        .map(move |y| {
            let width = ((r * r) as f32 - (y * y) as f32).sqrt() as isize;
            (-width..width).map(move |x| Coord(x, y))
        })
        .flatten()
}

// https://www.geeksforgeeks.org/bresenhams-circle-drawing-algorithm/
fn stroke_circle(r: isize) -> impl Iterator<Item = Coord> {
    let mut x = 0;
    let mut y = r;
    let mut d = 3 - 2 * r;

    std::iter::from_fn(move || {
        if y < x {
            return None;
        }

        let out = [
            Coord(x, y),
            Coord(-x, y),
            Coord(x, -y),
            Coord(-x, -y),
            Coord(y, x),
            Coord(-y, x),
            Coord(y, -x),
            Coord(-y, -x),
        ];

        x += 1;

        if d > 0 {
            y -= 1;
            d = d + 4 * (x - y) + 10;
        } else {
            d = d + 4 * x + 6;
        }

        Some(out)
    })
    .flatten()
}

/*
void drawCircle(int xc, int yc, int x, int y) {
}

// Function for circle-generation
// using Bresenham's algorithm
void circleBres(int xc, int yc, int r)
{
}
*/
