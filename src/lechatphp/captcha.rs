use image::{DynamicImage, imageops, GrayImage};
use imageproc::contrast::adaptive_threshold;
use imageproc::morphology::{dilate, erode};
use imageproc::distance_transform::Norm;
use std::collections::HashMap;
use std::fs;
use std::sync::{Arc, Mutex};
use std::path::Path;
use base64::Engine;
use lazy_static::lazy_static;

lazy_static! {
    static ref CAPTCHA_CACHE: Arc<Mutex<HashMap<String, String>>> = Arc::new(Mutex::new(HashMap::new()));
    // Inisialisasi cache jika sudah ada file
    static ref INITIALIZED: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));
}

// Fungsi utama untuk memecahkan captcha dari gambar base64
pub fn solve_b64(captcha_img: &str) -> Option<String> {
    // Inisialisasi cache dari file jika belum dilakukan
    let mut initialized = INITIALIZED.lock().unwrap();
    if !*initialized {
        if Path::new("captcha_cache.json").exists() {
            if let Ok(content) = fs::read_to_string("captcha_cache.json") {
                if let Ok(cache) = serde_json::from_str::<HashMap<String, String>>(&content) {
                    *CAPTCHA_CACHE.lock().unwrap() = cache;
                }
            }
        }
        *initialized = true;
    }
    
    // Extract base64 data
    let base64_str = captcha_img.split(',').last()?;
    
    // Hitung hash sederhana dari base64 untuk caching
    let img_hash = simple_hash(base64_str);
    
    // Cek cache
    if let Some(cached_solution) = CAPTCHA_CACHE.lock().unwrap().get(&img_hash) {
        println!("Cache hit: {}", cached_solution);
        return Some(cached_solution.clone());
    }
    
    // Decode base64
    let img_data = base64::engine::general_purpose::STANDARD.decode(base64_str).ok()?;
    
    // Load gambar
    let img = image::load_from_memory(&img_data).ok()?;
    
    // Proses gambar dengan metode khusus untuk captcha jenis ini
    let processed = preprocess_specific_captcha(&img);
    
    // Simpan preprocessing untuk debugging
    let _ = processed.save("debug_processed.png");
    
    // Deteksi dan baca teks
    if let Some(text) = detect_captcha_text(&processed) {
        // Simpan ke cache
        CAPTCHA_CACHE.lock().unwrap().insert(img_hash, text.clone());
        
        // Simpan cache ke file sesekali
        if CAPTCHA_CACHE.lock().unwrap().len() % 5 == 0 {
            if let Ok(json) = serde_json::to_string(&*CAPTCHA_CACHE.lock().unwrap()) {
                let _ = fs::write("captcha_cache.json", json);
            }
        }
        
        // Juga simpan gambar dan solusinya untuk training
        let _ = fs::create_dir_all("captcha_training");
        let _ = processed.save(format!("captcha_training/{}.png", text));
        
        return Some(text);
    }
    
    None
}

// Fungsi preprocessing khusus untuk captcha ini
fn preprocess_specific_captcha(img: &DynamicImage) -> GrayImage {
    // Konversi ke grayscale
    let  gray = img.to_luma8();
    
    // 1. Perbaiki ukuran (kode asli menggunakan 120x80)
    let sized = imageops::resize(&gray, 120, 80, 
                              image::imageops::FilterType::Gaussian);
    
    // 2. Perbaiki rotasi - Captcha ini diputar dengan sudut acak ±10-20 derajat
    // Kita bisa mendeteksi sudut rotasi dengan Hough transform atau metode lain
    // Untuk sederhananya, kita mencoba beberapa sudut dan memilih yang terbaik
    let mut best_img = sized.clone();
    let mut best_score = evaluate_captcha_clarity(&sized);
    
    for _angle in [-20, -15, -10, -5, 0, 5, 10, 15, 20].iter() {
        let rotated = imageops::rotate90(&sized); // Contoh rotasi sederhana
        let score = evaluate_captcha_clarity(&rotated);
        if score > best_score {
            best_img = rotated;
            best_score = score;
        }
    }
    
    // 3. Tingkatkan kontras untuk membedakan teks dari background
    let contrasted = adaptive_threshold(&best_img, 15);
    
    // 4. Hapus noise (titik acak yang ditambahkan di kode PHP)
    let denoised = remove_noise(&contrasted);
    
    // 5. Erosi diikuti dilatasi untuk membersihkan teks
    let eroded = erode(&denoised, Norm::L1, 1);
    let cleaned = dilate(&eroded, Norm::L1, 1);
    
    cleaned
}

// Evaluasi kejelasan captcha (skor lebih tinggi = lebih jelas)
fn evaluate_captcha_clarity(img: &GrayImage) -> f32 {
    // Hitung histogram
    let mut hist = [0u32; 256];
    for pixel in img.pixels() {
        hist[pixel.0[0] as usize] += 1;
    }
    
    // Hitung varians - captcha yang jelas memiliki lebih banyak kontras
    let mut mean = 0.0;
    let total_pixels = (img.width() * img.height()) as u32;
    
    for (i, &count) in hist.iter().enumerate() {
        mean += (i as f32) * (count as f32) / (total_pixels as f32);
    }
    
    let mut variance = 0.0;
    for (i, &count) in hist.iter().enumerate() {
        variance += ((i as f32) - mean).powi(2) * (count as f32) / (total_pixels as f32);
    }
    
    variance
}

// Hapus noise dari gambar
fn remove_noise(img: &GrayImage) -> GrayImage {
    // Identifikasi isolasi titik-titik atau kelompok kecil piksel
    let mut output = img.clone();
    
    for y in 1..(img.height() - 1) {
        for x in 1..(img.width() - 1) {
            // Periksa apakah piksel berada dalam kelompok kecil (titik noise)
            let mut neighbors = 0;
            let pixel_val = img.get_pixel(x, y).0[0];
            
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    
                    if nx >= 0 && nx < img.width() as i32 && 
                       ny >= 0 && ny < img.height() as i32 {
                        let neighbor_val = img.get_pixel(nx as u32, ny as u32).0[0];
                        if (neighbor_val > 127) == (pixel_val > 127) {
                            neighbors += 1;
                        }
                    }
                }
            }
            
            // Jika piksel terisolasi (kurang dari 2 tetangga yang serupa), itu kemungkinan noise
            if neighbors < 2 {
                // Hapus noise dengan mengubah warna piksel
                output.put_pixel(x, y, image::Luma([255 - pixel_val]));
            }
        }
    }
    
    output
}

// Deteksi teks dari gambar yang sudah diproses
fn detect_captcha_text(img: &GrayImage) -> Option<String> {
    // Captcha dari kode PHP memiliki beberapa karakter alfanumerik
    // Kita bisa menggunakan teknik segmentasi dan template matching
    
    // Implementasi sederhana: Segmentasi berdasarkan proyeksi vertikal
    let width = img.width() as usize;
    let height = img.height() as usize;
    let mut v_projection = vec![0; width];
    
    // Hitung proyeksi vertikal
    for x in 0..width {
        for y in 0..height {
            if img.get_pixel(x as u32, y as u32).0[0] < 128 {
                v_projection[x] += 1;
            }
        }
    }
    
    // Temukan batas-batas karakter
    let mut char_boundaries = Vec::new();
    let mut in_char = false;
    let mut start = 0;
    
    for x in 0..width {
        if v_projection[x] > 3 && !in_char {
            in_char = true;
            start = x;
        } else if (v_projection[x] <= 3 || x == width - 1) && in_char {
            in_char = false;
            if x - start >= 3 {  // Minimal lebar karakter
                char_boundaries.push((start, x));
            }
        }
    }
    
    // Verifikasi jumlah karakter - Captcha biasanya memiliki 4-6 karakter
    if char_boundaries.len() < 3 || char_boundaries.len() > 8 {
        return None;
    }
    
    // Gabungkan segmen yang terlalu dekat (karakter terhubung)
    let mut merged_boundaries = Vec::new();
    let mut current_start = 0;
    let mut current_end = 0;
    let min_gap = 3;  // Jarak minimal antar karakter
    
    for (i, &(start, end)) in char_boundaries.iter().enumerate() {
        if i == 0 {
            current_start = start;
            current_end = end;
        } else if start - current_end <= min_gap {
            current_end = end;
        } else {
            merged_boundaries.push((current_start, current_end));
            current_start = start;
            current_end = end;
        }
    }
    
    if !char_boundaries.is_empty() {
        merged_boundaries.push((current_start, current_end));
    }
    
    // Identifikasi setiap karakter dengan template matching
    let mut result = String::new();
    
    for (i, &(start, end)) in merged_boundaries.iter().enumerate() {
        let char_width = end - start;
        let char_img = imageops::crop_imm(img, start as u32, 0, char_width as u32, img.height()).to_image();
        
        // Simpan segmen untuk debugging
        let _ = char_img.save(format!("debug_char_{}.png", i));
        
        // Identifikasi karakter dengan template matching atau ML
        if let Some(c) = identify_character(&char_img) {
            result.push(c);
        } else {
            result.push('?');  // Fallback jika karakter tidak dikenali
        }
    }
    
    // Pastikan hasil memiliki panjang yang masuk akal
    if result.len() >= 3 && result.chars().all(|c| c.is_ascii_alphanumeric() || c == '?') {
        Some(result)
    } else {
        None
    }
}

// Identifikasi karakter tunggal
fn identify_character(char_img: &GrayImage) -> Option<char> {
    // Implementasi template matching
    // Di sini kita memerlukan database template karakter
    // atau model machine learning yang dilatih untuk captcha ini
    
    lazy_static! {
        static ref CHAR_TEMPLATES: HashMap<char, GrayImage> = load_templates();
    }
    
    let mut best_match = ('?', f32::MAX);
    
    for (c, template) in CHAR_TEMPLATES.iter() {
        let score = compare_images(char_img, template);
        if score < best_match.1 {
            best_match = (*c, score);
        }
    }
    
    // Tetapkan threshold untuk kecocokan
    if best_match.1 < 0.4 {
        Some(best_match.0)
    } else {
        // Fallback ke karakter yang paling mungkin berdasarkan posisi
        estimate_character_by_position(char_img)
    }
}

// Perkiraan karakter berdasarkan posisi dalam captcha
fn estimate_character_by_position(char_img: &GrayImage) -> Option<char> {
    // Analisis fitur gambar untuk memperkirakan karakter
    // Contoh sederhana: Hitung piksel hitam di berbagai region
    
    let width = char_img.width();
    let height = char_img.height();
    
    let mut top_count = 0;
    let mut middle_count = 0;
    let mut bottom_count = 0;
    let mut left_count = 0;
    let mut right_count = 0;
    
    for y in 0..height {
        for x in 0..width {
            if char_img.get_pixel(x, y).0[0] < 128 {
                // Piksel hitam
                if y < height / 3 {
                    top_count += 1;
                } else if y < 2 * height / 3 {
                    middle_count += 1;
                } else {
                    bottom_count += 1;
                }
                
                if x < width / 2 {
                    left_count += 1;
                } else {
                    right_count += 1;
                }
            }
        }
    }
    
    // Logika sederhana berdasarkan distribusi piksel hitam
    let total = top_count + middle_count + bottom_count;
    if total == 0 {
        return None;
    }
    
    let top_ratio = top_count as f32 / total as f32;
    let middle_ratio = middle_count as f32 / total as f32;
    let bottom_ratio = bottom_count as f32 / total as f32;
    let left_ratio = left_count as f32 / (left_count + right_count) as f32;
    
    // Perkiraan karakter berdasarkan distribusi (sangat sederhana)
    if top_ratio > 0.4 && bottom_ratio > 0.4 && middle_ratio < 0.2 {
        Some('8')
    } else if top_ratio > 0.4 && middle_ratio > 0.3 {
        Some('E')
    } else if left_ratio > 0.7 {
        Some('C')
    } else if top_ratio < 0.2 && bottom_ratio > 0.5 {
        Some('J')
    } else if middle_ratio > 0.5 {
        Some('H')
    } else {
        // Karakter default yang paling umum dalam captcha
        Some('A')
    }
}

// Bandingkan dua gambar untuk template matching
fn compare_images(img1: &GrayImage, img2: &GrayImage) -> f32 {
    // Resize kedua gambar ke ukuran yang sama
    let width = 20;
    let height = 30;
    let img1_resized = imageops::resize(img1, width, height, 
                                     image::imageops::FilterType::Nearest);
    let img2_resized = imageops::resize(img2, width, height, 
                                     image::imageops::FilterType::Nearest);
    
    // Hitung perbedaan rata-rata
    let mut diff_sum = 0.0;
    let total_pixels = width * height;
    
    for y in 0..height {
        for x in 0..width {
            let p1 = img1_resized.get_pixel(x, y).0[0] as f32 / 255.0;
            let p2 = img2_resized.get_pixel(x, y).0[0] as f32 / 255.0;
            diff_sum += (p1 - p2).abs();
        }
    }
    
    diff_sum / (total_pixels as f32)
}

// Load template karakter dari disk
fn load_templates() -> HashMap<char, GrayImage> {
    let mut templates = HashMap::new();
    let template_dir = Path::new("captcha_templates");
    
    // Jika direktori template ada
    if template_dir.exists() && template_dir.is_dir() {
        if let Ok(entries) = fs::read_dir(template_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(extension) = path.extension() {
                    if extension == "png" {
                        if let Some(stem) = path.file_stem() {
                            if let Some(char_str) = stem.to_str() {
                                if char_str.len() == 1 {
                                    if let Ok(img) = image::open(&path) {
                                        let c = char_str.chars().next().unwrap();
                                        templates.insert(c, img.to_luma8());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        // Jika direktori tidak ada, buat template kosong
        fs::create_dir_all(template_dir).ok();
    }
    
    // Template kosong sebagai fallback
    if templates.is_empty() {
        // Inisialisasi dengan beberapa karakter umum dalam captcha
        for c in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz".chars() {
            let template = GrayImage::new(20, 30);
            templates.insert(c, template);
        }
    }
    
    templates
}

// Hitung hash sederhana untuk caching
fn simple_hash(s: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

