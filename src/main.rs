use std::{fs, thread};
use rand::Rng;
use std::time::{Instant};
use image::{GrayImage, Luma};
use std::sync::{Arc, Mutex};
use std::fs::File;
use std::io::Write;
use std::io::BufWriter;
use std::io::Read;

// Do not adjust these parameters unless you aren't using the MNIST data
const INPUT_FEATURES: usize = 28 * 28; // 28x28 images
const OUTPUT_SIZE: usize = 10; // ten digits

// These parameters are adjustable
const LOAD_MODEL_FROM_FILE: bool = false; // set it to false to train from scratch
const SAVE_MODEL_TO_FILE: bool = true; // set it to true to save the weights to model-weights.bin
const EPOCHS: usize = 10; // how many times the model is trained on all the training images
const HIDDEN_LAYER_COUNT: usize = 2; // total amount of hidden layers
const HIDDEN_LAYER_SIZE: usize = 300; // perceptrons per each hidden layer
const LEARNING_RATE: f64 = 7e-3; // Learning rate of the model
const DROPOUT_RATE: f64 = 0.25; // chance that a neuron gets dropped during each batch
const BATCH_SIZE: usize = 10; // batch size for training
const LEAKY_RELU_SLOPE: f64 = 0.01; // slope of the leaky ReLU function for x < 0
const PARALLEL_THREAD_CT: usize = 16; // how many threads are used in parallel to increase computation speed for some functions

fn main() {
	if HIDDEN_LAYER_COUNT == 0 {
		panic!("There must be at least one hidden layer!");
	}

	let cpu_count = num_cpus::get();

	println!("Detected {} logical CPUs on this machine. Using {} threads for parallelization", cpu_count, PARALLEL_THREAD_CT);

	let mut model: Vec<Vec<Vec<f64>>> = vec![new_matrix(INPUT_FEATURES+1, HIDDEN_LAYER_SIZE, "random uniform")];
	let no_model_dropout = vec![new_matrix(1, HIDDEN_LAYER_SIZE, "one"); HIDDEN_LAYER_COUNT]; // mask for full model
	if HIDDEN_LAYER_COUNT > 1 {
		for _ in 1..HIDDEN_LAYER_COUNT {
			model.push(new_matrix(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, "random uniform"));
		}
	}

	model.push(new_matrix(HIDDEN_LAYER_SIZE, OUTPUT_SIZE, "random uniform"));

	if LOAD_MODEL_FROM_FILE {
		// first load model weights
		println!("Loading model from file...");
		let mut file = File::open("model-weights.bin").expect("Could not open file! Maybe it does not exist");
		for i in 0..model.len() {
			for j in 0..model[i].len() {
				for k in 0..model[i][j].len() {
					// an f64 is 8 bytes
					let mut this_float_le = [0; 8];
					let count = file.read(&mut this_float_le).expect("Could not read file");
					if count < 8 {
						panic!("Could not read full 8 bytes! Most likely the current model parameters are different than the ones which were used in the creation of the model-weights.bin file");
					}
					model[i][j][k] = f64::from_le_bytes(this_float_le);
				}
			}
		}

		println!("Loading image...");

		// then evaluate on a test image
		let grey_img = image::open("test-image.png")
			.expect("Could not open image file. Most likely it does not exist")
			.into_luma8();

		let mut image_vec: Vec<f64> = vec![1.];
		for i in 0..INPUT_FEATURES {
			let p = grey_img.get_pixel((i % 28) as u32, (i / 28) as u32);
			image_vec.push(p[0] as f64 / 255.);
		}

		let prediction = get_model_predictions(&model, &no_model_dropout, &vec![image_vec]);
		println!("The model's prediction is: {}", prediction[0]);

		return
	}

	println!("Loading training images...");
	let mut train_imgs = load_mnist_images_file("./data/train-images-idx3-ubyte");
	println!("Loading training labels...");
	let mut train_lbls = load_mnist_labels_file("./data/train-labels-idx1-ubyte");

	println!("Loading testing images...");
	let test_imgs = load_mnist_images_file("./data/t10k-images-idx3-ubyte");
	println!("Loading testing labels...");
	let test_lbls = load_mnist_labels_file("./data/t10k-labels-idx1-ubyte");

	println!("Data stats:\nTraining images: {}\nTraining labels: {}\nTesting images: {}\nTesting labels: {}", train_imgs.len(), train_lbls.len(), test_imgs.len(), test_lbls.len());

	assert!(train_imgs.len() == train_lbls.len(), "Different amount of training images and labels");
	assert!(test_imgs.len() == test_lbls.len(), "Different amount of testing images and labels");
	assert!(train_imgs.len() % BATCH_SIZE == 0, "Non-integer amount of batches for training");
	assert!(test_imgs.len() % BATCH_SIZE == 0, "Non-integer amount of batches for testing");

	println!("Randomizing data order...");
	randomize_sample_order(&mut train_imgs, &mut train_lbls);

	// uncomment to save an image from the training set to disk
	// let test_i = 12345;
	// let test_img = train_imgs[test_i][1..].to_vec();
	// preview_img(&test_img);
	// println!("Training image {} has label {}", test_i, train_lbls[test_i]);

	println!("Starting training...");
	for e in 1..=EPOCHS {
		println!("Starting epoch {}...", e);
		// train the model first
		let mut correct_count: usize = 0;
		let mut start_inst = Instant::now();
		for i in 0..(train_imgs.len() / BATCH_SIZE) {
			// generate images and labels matrices
			let mut imgs_matrix = train_imgs[i*BATCH_SIZE..((i+1)*BATCH_SIZE)].to_owned();
			let mut lbls_matrix: Vec<Vec<f64>> = vec![new_one_hot_label(train_lbls[i*BATCH_SIZE] as usize, OUTPUT_SIZE)];
			for j in 1..BATCH_SIZE {
				lbls_matrix.push(new_one_hot_label(train_lbls[i*BATCH_SIZE+j] as usize, OUTPUT_SIZE))
			}

			// generate dropout vectors
			let mut dropout_vectors: Vec<Vec<Vec<f64>>> = Vec::new();
			// we don't want to touch the output layer
			for _ in 0..HIDDEN_LAYER_COUNT {
				dropout_vectors.push(new_matrix(BATCH_SIZE, HIDDEN_LAYER_SIZE, "Bernoulli"));
			}

			// do forward pass and check correctness
			let predictions = get_model_predictions(&model, &dropout_vectors, &imgs_matrix);
			for j in (0..BATCH_SIZE).rev() {
				if predictions[j] == train_lbls[i*BATCH_SIZE+j] as usize {
					correct_count += 1;
					lbls_matrix.remove(j);
					imgs_matrix.remove(j);
					for k in 0..dropout_vectors.len() {
						dropout_vectors[k].remove(j);
					}
				}
			}

			if imgs_matrix.len() > 0 {
				// do the forward pass again with images the model identified wrongly
				let cache = model_forward_pass(&model, &dropout_vectors, &imgs_matrix);

				//do backprop pass
				model_backprop_pass(&mut model, &dropout_vectors, &cache, &lbls_matrix);
			}

			if i % 10 == 0 && start_inst.elapsed().as_secs() >= 2 {
				println!("Trained on {} out of {} batches so far", i, train_imgs.len() / BATCH_SIZE);
				start_inst = Instant::now();
			}
		}
		println!("Training accuracy: {} out of {}", correct_count, train_imgs.len());
		println!("Validating model...");

		// validate the model
		correct_count = 0;
		for i in 0..test_imgs.len() {
			if get_model_predictions(&model, &no_model_dropout, &vec![test_imgs[i].clone()])[0] == test_lbls[i] as usize {
				correct_count += 1;
			}
		}
		println!("Validation accuracy: {} out of {}", correct_count, test_imgs.len());
		println!("");
	}

	if !SAVE_MODEL_TO_FILE { return };

	// save model weights
	println!("Saving model weights...");
	let file = File::create("model-weights.bin").expect("Failed to create file to store model weights");
	let mut file = BufWriter::new(file);
	for i in 0..model.len() {
		for j in 0..model[i].len() {
			for k in 0..model[i][j].len() {
				file.write(&model[i][j][k].to_le_bytes()).expect("Could not write to file");
			}
		}
	}
}

fn load_mnist_images_file(file_path: &str) -> Vec<Vec<f64>> {
	let raw_data: Vec<u8> = fs::read(file_path).expect("Failed to open file! Most likely the file path is incorrectly set");

	if raw_data[2] != 8 || raw_data[3] != 3 {
		panic!("Not an IDX images file, based on magic number");
	}

	let mut image_vecs: Vec<Vec<f64>> = vec![vec![1.]];

	let mut current_img = 0;
	for i in 16..raw_data.len() {
		if (i - 16) % INPUT_FEATURES == 0 && i > 16 {
			image_vecs.push(vec![1.]);
			current_img += 1;
		}
		image_vecs[current_img].push((raw_data[i] as f64) / 255.); // normalize pixel values from [0, 255] to [0, 1]
	}

	image_vecs
}

fn load_mnist_labels_file(file_path: &str) -> Vec<u8> {
	let raw_data: Vec<u8> = fs::read(file_path).expect("Failed to open file! Most likely the file path is incorrectly set");

	if raw_data[2] != 8 || raw_data[3] != 1 {
		panic!("Not an IDX labels file, based on magic number");
	}

	raw_data[8..].to_vec()
}

fn randomize_sample_order(samples: &mut Vec<Vec<f64>>, labels: &mut Vec<u8>) {
	assert!(samples.len() == labels.len(), "Arrays of samples and labels have different lengths");
	for i in 0..(samples.len()-1) {
		let j: usize = rand::random_range(0..(samples.len()-1));
		samples.swap(i, j);
		labels.swap(i, j);
	}
}

fn preview_img(img_vec: &Vec<f64>) {
	assert!(img_vec.len() == 28*28, "Image vector needs to have a length of 784");
	let mut img = GrayImage::new(28, 28);
	for i in 0..784 {
		img.put_pixel(i % 28, (i / 28) as u32, Luma([(img_vec[i as usize]*255.).round() as u8]))
	}
	img.save("img-preview.png").unwrap();
}

fn new_matrix(rows: usize, cols: usize, initialization_method: &str) -> Vec<Vec<f64>> {
	if rows == 0 || cols == 0 {
		panic!("New matrix has zero rows or columns");
	}

	if initialization_method == "zero" {
		return vec![vec![0.; cols]; rows];
	} else if initialization_method == "one" {
		return vec![vec![1.; cols]; rows];
	}

	let mut rng = rand::rng();
	let mut matrix: Vec<Vec<f64>> = Vec::new();
	for i in 0..rows {
		matrix.push(Vec::new());
		for _ in 0..cols {
			if initialization_method == "random uniform" {
				// uniform distribution between -1 and 1
				matrix[i].push(2. * rng.random::<f64>() - 1.);
			} else if initialization_method == "Bernoulli" {
				let random = rng.random::<f64>();
				if random < DROPOUT_RATE {
					matrix[i].push(0.);
				} else {
					// use  1 / (1-p) instead of 1 since we're using inverse dropout
					matrix[i].push(1. / (1. - DROPOUT_RATE));
				}
			} else {
				panic!("Unknown initialization method {}", initialization_method);
			}
		}
	}
	matrix
}

fn matrix_transpose(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
	// rows and columns of new matrix
	let r = matrix[0].len();
	let c = matrix.len();

	let mut transpose = new_matrix(r, c, "zero");
	for i in 0..r {
		for j in 0..c {
			transpose[i][j] = matrix[j][i];
		}
	}

	transpose
}

fn matrix_subtract(mat_a: &Vec<Vec<f64>>, mat_b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
	let ra = mat_a.len(); // rows of A
	let ca = mat_a[0].len(); // cols of B
	let rb = mat_b.len();
	let cb = mat_b[0].len();

	if ra != rb || ca != cb {
		panic!(
			"Matrix sizes are incompatible for subtraction: {}x{} and {}x{}",
		 ra, ca, rb, cb
		);
	}

	// define Arc matrices Arc for use with parallelization
	let mat_a_arc = Arc::new(mat_a.clone());
	let mat_b_arc = Arc::new(mat_b.clone());

	let thread_count = if PARALLEL_THREAD_CT > ra { ra } else { PARALLEL_THREAD_CT };
	let result: Arc<Mutex<Vec<Vec<Vec<f64>>>>> = Arc::new(Mutex::new(vec![Vec::new(); thread_count]));
	let mut handles = Vec::new();

	for i in 0..thread_count {
		let mat_a1 = Arc::clone(&mat_a_arc);
		let mat_b1 = Arc::clone(&mat_b_arc);
		let result = Arc::clone(&result);
		let start_row = ((ra as f64) / (thread_count as f64) * (i as f64)).floor() as usize;
		let end_row = ((ra as f64) / (thread_count as f64) * ((i+1) as f64)).floor() as usize;
		let handle = thread::spawn(move || {
			let mut partial_result = new_matrix(end_row - start_row, cb, "zero");
			// row iterator
			for j in start_row..end_row {
				// col iterator
				for k in 0..ca {
					partial_result[j - start_row][k] = mat_a1[j][k] - mat_b1[j][k];
				}
			}
			result.lock().unwrap()[i] = partial_result;
		});

		handles.push(handle);
	}

	for handle in handles {
		handle.join().unwrap();
	}

	result.lock().unwrap().concat()
}

fn matrix_multiply(mat_a: &Vec<Vec<f64>>, mat_b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
	let ra = mat_a.len(); // rows of A
	let ca = mat_a[0].len(); // cols of B
	let rb = mat_b.len();
	let cb = mat_b[0].len();

	if ca != rb {
		panic!(
			"Matrix sizes are incompatible for multiplication: {}x{} and {}x{}",
			ra, ca, rb, cb
		);
	}

	// define Arc matrices for use with parallelization
	let mat_a_arc = Arc::new(mat_a.clone());
	let mat_b_arc = Arc::new(mat_b.clone());

	let thread_count = if PARALLEL_THREAD_CT > ra { ra } else { PARALLEL_THREAD_CT };
	let prod: Arc<Mutex<Vec<Vec<Vec<f64>>>>> = Arc::new(Mutex::new(vec![Vec::new(); thread_count]));
	let mut handles = Vec::new();

	for i in 0..thread_count {
		let mat_a1 = Arc::clone(&mat_a_arc);
		let mat_b1 = Arc::clone(&mat_b_arc);
		let prod = Arc::clone(&prod);
		let start_row = ((ra as f64) / (thread_count as f64) * (i as f64)).floor() as usize;
		let end_row = ((ra as f64) / (thread_count as f64) * ((i+1) as f64)).floor() as usize;
		let handle = thread::spawn(move || {
			let mut partial_prod = new_matrix(end_row - start_row, cb, "zero");
			// row iterator
			for j in start_row..end_row {
				// col iterator
				for k in 0..cb {
					// cross product iterator
					for l in 0..ca {
						partial_prod[j - start_row][k] += mat_a1[j][l] * mat_b1[l][k]
					}
				}
			}
			prod.lock().unwrap()[i] = partial_prod;
		});

		handles.push(handle);
	}

	for handle in handles {
		handle.join().unwrap();
	}

	prod.lock().unwrap().concat()
}

fn matrix_scalar_mult(scalar: &f64, mat: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
	let mut result = new_matrix(mat.len(), mat[0].len(), "zero");
	for i in 0..mat.len() {
		for j in 0..mat[0].len() {
			result[i][j] = mat[i][j] * scalar;
		}
	}

	result
}

// the hadamard product is just element-wise multiplication on matrices
fn matrix_hadamard(mat_a: &Vec<Vec<f64>>, mat_b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
	let ra = mat_a.len(); // rows of A
	let ca = mat_a[0].len(); // cols of B
	let rb = mat_b.len();
	let cb = mat_b[0].len();

	if ra != rb || ca != cb {
		panic!(
			"Matrix sizes are incompatible for Hadamard product: {}x{} and {}x{}",
			ra, ca, rb, cb
		);
	}

	let mut result = new_matrix(ra, ca, "zero");

	for i in 0..ra {
		for j in 0..ca {
			result[i][j] = mat_a[i][j] * mat_b[i][j];
		}
	}

	result
}
// leaky ReLU function: x for x >= 0 and 0.01x for x < 0
fn relu_of_matrix(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
	let mut result = new_matrix(matrix.len(), matrix[0].len(), "zero");
	for i in 0..(matrix.len()) {
		for j in 0..(matrix[0].len()) {
			if matrix[i][j] >= 0. {
				result[i][j] = matrix[i][j];
			} else {
				result[i][j] = matrix[i][j] * LEAKY_RELU_SLOPE;
			}
		}
	}
	result
}

// derivative of the leaky ReLU
fn relu_prime_of_matrix(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
	let mut result = new_matrix(matrix.len(), matrix[0].len(), "zero");
	for i in 0..(matrix.len()) {
		for j in 0..(matrix[0].len()) {
			if matrix[i][j] >= 0. {
				result[i][j] = 1.;
			} else {
				result[i][j] = LEAKY_RELU_SLOPE;
			}
		}
	}
	result
}

// row-wise softmax operation
fn softmax_of_matrix(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
	let r = matrix.len();
	let c = matrix[0].len();

	let mut softmax = new_matrix(r, c, "zero");

	// subtract the largest value of each row from its elements
	for i in 0..r {
		// find largest value
		let mut largest: f64 = 0.;
		for j in 0..c {
			if matrix[i][j] > largest || largest == 0. {
				largest = matrix[i][j];
			}
		}

		// subtract it
		for j in 0..c {
			softmax[i][j] = matrix[i][j] - largest;
		}
	}

	for i in 0..r {
		// get sum of exp's
		let mut exp_sum: f64 = 0.;
		for j in 0..c {
			exp_sum += std::f64::consts::E.powf(softmax[i][j]);
		}

		// calculate softmax per-element
		for j in 0..c {
			softmax[i][j] = std::f64::consts::E.powf(softmax[i][j]) / exp_sum;
		}
	}

	softmax
}

fn new_one_hot_label(position: usize, total_size: usize) -> Vec<f64> {
	let mut vector = vec![0.; total_size];
	vector[position] = 1.;
	vector
}

fn model_forward_pass(model: &Vec<Vec<Vec<f64>>>,
	dropout_vectors: &Vec<Vec<Vec<f64>>>,
	input_sample: &Vec<Vec<f64>>) -> Vec<Vec<Vec<f64>>> {
	let mut intermediate_steps = vec![input_sample.to_owned(); model.len()+1];
	for i in 0..model.len() {
		// doing this operation: (s(R*W)) * d
		// where s is the element-wise sigmoid, R is the output from the previous layer, W is layer-to-layer weights
		// and d is the dropout vector
		intermediate_steps[i+1] = relu_of_matrix(
			&matrix_multiply(&intermediate_steps[i], &model[i]),
		);

		// do dropout on all the outputs except for the final one
		if i < model.len() - 1 {
			// do element-wise multiplication of the output of this layer with the dropout vector
			intermediate_steps[i+1] = matrix_hadamard(&dropout_vectors[i], &intermediate_steps[i+1])
		}
	}
	// perform softmax on the last step to get probabilities
	intermediate_steps.push(softmax_of_matrix(&intermediate_steps[intermediate_steps.len()-1]));
	intermediate_steps
}

// convenience function to do a forward pass and get the model's prediction
// returns a vector with the prediction of the model for each input row
fn get_model_predictions(model: &Vec<Vec<Vec<f64>>>,
	dropout_vectors: &Vec<Vec<Vec<f64>>>,
	input_sample: &Vec<Vec<f64>>) -> Vec<usize> {
	let cache = model_forward_pass(model, dropout_vectors, input_sample);
	let softmax = &cache[cache.len()-1];
	let mut output: Vec<usize> = Vec::new();

	for i in 0..softmax.len() {
		// find the best guess
		let mut best_guess_i = 0;
		let mut best_pct: f64 = 0.;
		for j in 0..softmax[i].len() {
			let p = softmax[i][j];
			if !p.is_finite() {
				panic!("Encountered a NaN or infinite value! Please decrease the learning rate");
			}
			if p > best_pct {
				best_pct = p;
				best_guess_i = j;
			}
		}
		output.push(best_guess_i)
	}
	return output;
}

fn model_backprop_pass(model: &mut Vec<Vec<Vec<f64>>>,
	dropout_vectors: &Vec<Vec<Vec<f64>>>,
	intermediate_steps: &Vec<Vec<Vec<f64>>>,
	sample_label: &Vec<Vec<f64>>) {
	// batch size can be taken from the amount of rows of the input
	let batch_size = intermediate_steps[0].len();
	// how much to change the weights
	// calculation is: (O^T - L) * s'(H)
	// where O is the output matrix (w/ softmax), L is the sample label, s' is sigmoid prime, and H is the last hidden layer output
	let mut delta_t = matrix_hadamard(
		&matrix_subtract(
			&intermediate_steps[intermediate_steps.len()-1],
			sample_label
		),
		&relu_prime_of_matrix(&intermediate_steps[intermediate_steps.len()-2])
	);

	for i in 1..=model.len() {
		// first access index, for model weights
		let i1 = model.len() - i;
		// 2nd index, for the previous layer activation in `intermediate_steps`
		let i2 = intermediate_steps.len() - i - 2;

		model[i1] = matrix_subtract(
			&model[i1],
			&matrix_scalar_mult(
				&(LEARNING_RATE / batch_size as f64),
				&matrix_multiply(
					&matrix_transpose(&intermediate_steps[i2]),
					&delta_t
				)
			)
		);

		if i1 > 0 {
			delta_t = matrix_hadamard(&matrix_hadamard(
				&relu_prime_of_matrix(&intermediate_steps[i2]),
				&matrix_multiply(
					&delta_t,
					&matrix_transpose(&model[i1])
				)
			), &dropout_vectors[i1 - 1]);
		}
	}
}
