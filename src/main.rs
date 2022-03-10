use anyhow::{Context, Result};
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
    #[structopt(short, long, default_value = "10000")]
    n_points: usize,

    /// Minimum point size
    #[structopt(long = "minp", default_value = "3")]
    min_point_size: isize,

    /// Minimum point size
    #[structopt(long = "maxp", default_value = "10")]
    max_point_size: isize,

    /// Random seed (unsigned integer)
    #[structopt(short, long)]
    seed: Option<u64>,
}

fn main() -> Result<()> {
    let args = Opt::from_args();

    let input_img = read_png(&args.input).context("Failed to read input image")?;

    let mut output_img = RgbImage::new(input_img.width(), input_img.height());

    let center = Coord(output_img.width() as isize / 2, output_img.height() as isize / 2);

    let r = output_img.width().min(output_img.height()) as isize / 2;
    for coord in fill_circle(r).map(|c| center + c) {
        output_img.set(coord, [0xff, 0x00, 0x00]);
    }

    write_png(&args.output, &output_img)
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

    pub fn get(&self, coord: Coord) -> [u8; 3] {
        let idx = self.calc_index(coord).unwrap();
        [self.data[idx + 0], self.data[idx + 1], self.data[idx + 2]]
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
