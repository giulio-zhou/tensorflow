// #![allow(dead_code)]
extern crate time;
extern crate byteorder;
extern crate libc;

use std::net::{TcpListener, TcpStream};
// use byteorder::{LittleEndian, WriteBytesExt, ReadBytesExt};
use std::io::{Read, Write};
use std::mem;
use std::slice;
use std::time::Duration;
// use std::thread;
use std::str;
use std::ffi::CStr;
use libc::c_char;
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::collections::HashMap;
use std::sync::atomic::{AtomicPtr, Ordering};

// const SHUTDOWN_CODE: u8 = 0;
// const FIXEDINT_CODE: u8 = 1;
// const FIXEDFLOAT_CODE: u8 = 2;
// const FIXEDBYTE_CODE: u8 = 3;
// const VARINT_CODE: u8 = 4;
// const VARFLOAT_CODE: u8 = 5;
// const VARBYTE_CODE: u8 = 6;
// const STRING_CODE: u8 = 7;
//
// must be a multiple of mem_size_of::<f64>()
const HEADER_LENGTH_BYTES: usize = 2 * 8;

enum BatchStatus {
    Read, // input buffer can be read into
    Write(u32), // output buffer can be written to
    Compute, // Do nothing
}

struct Buffer {
    ptr: AtomicPtr<u8>,
    pos: usize,
    len: usize,
    capacity: usize,
}

impl Buffer {
    pub fn new(raw: (*mut f64, usize)) -> Buffer {
        let (ptr, capacity) = raw;
        Buffer {
            ptr: AtomicPtr::new(ptr as *mut u8),
            pos: 0,
            len: 0,
            capacity: capacity * mem::size_of::<f64>(),
        }
    }

    pub fn set_length(&mut self, length: usize) {
        assert!(length <= self.capacity);
        self.len = length;
    }

    pub fn get_length(&self) -> usize {
        self.len
    }

    #[allow(dead_code)]
    pub fn get_capacity(&self) -> usize {
        self.capacity
    }

    pub fn set_pos(&mut self, pos: usize) {
        assert!(pos <= self.len);
        self.pos = pos;
    }

    /// increase position by
    pub fn increment_pos(&mut self, increment: usize) {
        assert!(self.pos + increment <= self.len);
        self.pos += increment;
    }

    pub fn get_pos(&self) -> usize {
        self.pos
    }


    pub fn get_slice(&self) -> &[u8] {
        let s: &[u8] = unsafe {
            let ptr = self.ptr.load(Ordering::Relaxed);
            assert!(!ptr.is_null());
            let slice: &[u8] = slice::from_raw_parts(ptr, self.capacity);
            &slice[self.pos..self.len]
        };
        s
    }

    pub fn get_slice_mut(&mut self) -> &mut [u8] {
        let s: &mut [u8] = unsafe {
            let ptr = self.ptr.load(Ordering::Relaxed);
            assert!(!ptr.is_null());
            let slice: &mut [u8] = slice::from_raw_parts_mut(ptr, self.capacity);
            &mut slice[self.pos..self.len]
        };
        s
    }
}


struct Batch {
    recv_buffer: Buffer,
    send_buffer: Buffer,
    read_start_time: u64,
    status: Arc<Mutex<BatchStatus>>,
    id: u32,
}

impl Batch {
    pub fn new(id: u32, recv_buffer: Buffer, send_buffer: Buffer) -> Batch {
        Batch {
            recv_buffer: recv_buffer,
            send_buffer: send_buffer,
            read_start_time: 0,
            status: Arc::new(Mutex::new(BatchStatus::Read)),
            id: id,
        }

    }
}

// /// Compute the percentile rank of `snapshot`. The rank `p` must
// /// be in [0.0, 1.0] (inclusive) and `snapshot` must be sorted.
// ///
// /// Algorithm is the third variant from
// /// [Wikipedia](https://en.wikipedia.org/wiki/Percentile)
// fn percentile(snapshot: &Vec<i64>, p: f64) -> f64 {
//     assert!(snapshot.len() > 0);
//     let sample_size = snapshot.len() as f64;
//     assert!(p >= 0.0 && p <= 1.0, "percentile out of bounds");
//     let x = if p <= 1.0 / (sample_size + 1.0) {
//         // println!("a");
//         1.0
//     } else if p > 1.0 / (sample_size + 1.0) && p < sample_size / (sample_size + 1.0) {
//         // println!("b");
//         p * (sample_size + 1.0)
//     } else {
//         // println!("c");
//         sample_size
//     };
//     let index = x.floor() as usize - 1;
//     let v = snapshot[index] as f64;
//     let rem = x % 1.0;
//     let per = if rem != 0.0 {
//         // println!("rem: {}", rem);
//         v + rem * (snapshot[index + 1] - snapshot[index]) as f64
//     } else {
//         v
//     };
//     per
// }


fn compute_stats(snap: &Vec<u64>) -> (f64, f64, f64) {
    let mean = snap.iter().fold(0, |acc, &x| acc + x) as f64 / snap.len() as f64;
    let mut var: f64 = snap.iter().fold(0.0, |acc, &x| acc + (x as f64 - mean).powi(2));
    var = var / (snap.len() - 1) as f64;
    let mut snap_cpy = snap.to_vec();
    snap_cpy.sort();
    let index = (snap_cpy.len() as f64) * 0.99;
    let p99 = snap_cpy[index as usize] as f64;
    (mean, var, p99)
}


fn monitor_timing(read_times: Arc<Mutex<Vec<u64>>>, queue_times: Arc<Mutex<Vec<u64>>>) {
    loop {
        let report_interval_secs = 20;
        thread::sleep(Duration::new(report_interval_secs, 0));
        let rt_snap = {
            let mut rt_inner = read_times.lock().unwrap();
            let rt_snap = rt_inner.clone();
            // rt_inner.clear();
            rt_snap
        };
        let qt_snap = {
            let mut qt_inner = queue_times.lock().unwrap();
            let qt_snap = qt_inner.clone();
            // qt_inner.clear();
            qt_snap
        };
        let (rt_mean, rt_var, rt_p99) = compute_stats(&rt_snap);
        let (qt_mean, qt_var, qt_p99) = compute_stats(&qt_snap);
        println!("Read (ms): mean {}, std {}, p99 {}",
                 rt_mean / 1000.0,
                 rt_var.sqrt() / 1000.0,
                 rt_p99 / 1000.0);
        println!("Queue (ms): mean {}, std {}, p99 {}",
                 qt_mean / 1000.0,
                 qt_var.sqrt() / 1000.0,
                 qt_p99 / 1000.0);
        // rt_snap.sort();
        // qt_snap.sort();

    }

}

fn socket_event_loop_run(mut stream: TcpStream,
                         mut batches: Vec<Batch>,
                         batch_ready: mpsc::Sender<(Header, u64, u64)>) {

    let num_batches = batches.len();
    let mut cur_reading_batch = 0;
    let mut cur_writing_batch = 0;
    for b in batches.iter_mut() {
        b.recv_buffer.set_length(HEADER_LENGTH_BYTES);
    }
    // TODO figure out when the client has hung up and break loop
    loop {
        let mut i = 0;
        for b in batches.iter_mut() {
            let mut status = b.status.lock().unwrap();
            match *status {
                BatchStatus::Read => {
                    // Try to read input buffer but don't block
                    // If we finish reading, set status to wait
                    if i == cur_reading_batch {
                        let maybe_read_start = time::precise_time_ns();
                        // let mut read_buffer = b.recv_buffer.unlock().unwrap();
                        let cur_max_read_size =
                            (b.recv_buffer.get_length() - b.recv_buffer.get_pos()) as u64;
                        // println!("recv_buffer_length: {}, recv_buffer_pos: {}",
                        //          b.recv_buffer.get_length(),
                        //          b.recv_buffer.get_pos());
                        let mut cur_batch_handle = stream.try_clone()
                            .unwrap()
                            .take(cur_max_read_size);
                        let bytes_read =
                            match cur_batch_handle.read(b.recv_buffer.get_slice_mut()) {
                                Ok(bytes_read) => bytes_read,
                                Err(_) => 0,
                            };
                        // if bytes_read > 0 {
                        //     println!("Bytes read: {}", bytes_read);
                        //     assert!(b.recv_buffer.get_pos() + bytes_read <=
                        //             b.recv_buffer.get_length());
                        // }
                        // parse header to update length

                        // READ STARTED
                        if b.recv_buffer.get_pos() == 0 && bytes_read > 0 {
                            b.read_start_time = maybe_read_start;
                        }
                        if b.recv_buffer.get_pos() == 0 && bytes_read >= HEADER_LENGTH_BYTES {
                            // parse header
                            let header_slice: &[u32] = unsafe {
                                slice::from_raw_parts(b.recv_buffer.get_slice().as_ptr() as *const u32,
                                    HEADER_LENGTH_BYTES / mem::size_of::<u32>())
                            };
                            // let code = header_slice[0];
                            let num_inputs = header_slice[1];
                            let input_len = header_slice[2];
                            // println!("Read header: {:?}, code: {}, num_inputs: {}, input_len: {}",
                            //          header_slice,
                            //          code,
                            //          num_inputs,
                            //          input_len);

                            let current_batch_length = HEADER_LENGTH_BYTES as u32 +
                                                       num_inputs * input_len *
                                                       mem::size_of::<f64>() as u32;
                            b.recv_buffer.set_length(current_batch_length as usize);
                        }
                        if bytes_read > 0 {
                            b.recv_buffer.increment_pos(bytes_read);
                        }

                        // Done reading batch
                        if b.recv_buffer.get_pos() == b.recv_buffer.get_length() &&
                           b.recv_buffer.get_length() >= HEADER_LENGTH_BYTES {
                            let read_end_time = time::precise_time_ns();
                            let queue_start_time = time::precise_time_ns();


                            // set the position back to 0 for reading again
                            b.recv_buffer.set_pos(0);
                            // set the next batch as ready to read
                            cur_reading_batch = (cur_reading_batch + 1) % num_batches;

                            // update status
                            *status = BatchStatus::Compute;

                            let header_slice: &[u32] = unsafe {
                                slice::from_raw_parts(b.recv_buffer.get_slice().as_ptr() as *const u32,
                                    HEADER_LENGTH_BYTES / mem::size_of::<u32>())
                            };
                            let code = header_slice[0];
                            let num_inputs = header_slice[1];
                            let input_len = header_slice[2];

                            let h = Header {
                                batch_id: b.id,
                                code: code,
                                num_inputs: num_inputs,
                                input_len: input_len,
                            };

                            // println!("Sending batch header: {:?}", h);
                            // reset length to just header length
                            b.recv_buffer.set_pos(0);
                            b.recv_buffer.set_length(HEADER_LENGTH_BYTES);
                            let read_time = read_end_time - b.read_start_time;
                            batch_ready.send((h, read_time, queue_start_time)).unwrap();
                        }
                    }
                }
                BatchStatus::Compute => {
                    // do nothing
                }
                BatchStatus::Write(num_bytes) => {
                    // Try to send
                    // track the order of the writing batches because we
                    // want to ensure in-order delivery of responses to the
                    // batches
                    if i == cur_writing_batch {
                        b.send_buffer.set_length(num_bytes as usize);
                        let bytes_written = stream.write(b.send_buffer.get_slice()).unwrap();
                        assert!(bytes_written + b.send_buffer.get_pos() <=
                                b.send_buffer.get_length());
                        b.send_buffer.increment_pos(bytes_written);
                        // Check if we're done writing
                        if b.send_buffer.get_pos() == b.send_buffer.get_length() {
                            // ready to write next batch when it's ready
                            cur_writing_batch = (cur_writing_batch + 1) % num_batches;
                            // ready to read again into this batch
                            *status = BatchStatus::Read;
                            b.send_buffer.set_pos(0);
                        }
                    }
                }
            }
            i += 1;
        }
    }
}


#[repr(C)]
#[derive(Clone, Debug)]
pub struct Header {
    pub batch_id: u32,
    pub code: u32,
    pub num_inputs: u32,
    pub input_len: u32,
}


struct Connection {
    // batches: Vec<Arc<Batch>>,
    // first u64 is read_time, second is queue_start_time
    batch_receiver: mpsc::Receiver<(Header, u64, u64)>,
    batch_status: HashMap<u32, Arc<Mutex<BatchStatus>>>,
    current_batch: Option<Header>,
    #[allow(dead_code)]
    event_loop_handle: JoinHandle<()>,
}

pub struct BufferedServer {
    listener: TcpListener,
    read_times: Arc<Mutex<Vec<u64>>>,
    queue_times: Arc<Mutex<Vec<u64>>>,
    connection: Option<Connection>, /* stream: Option<TcpStream>,
                                     * header: Option<Header>, */
}

impl BufferedServer {
    pub fn new(address: &str) -> BufferedServer {

        let mw = BufferedServer {
            listener: TcpListener::bind(address).unwrap(),
            read_times: Arc::new(Mutex::new(Vec::new())),
            queue_times: Arc::new(Mutex::new(Vec::new())),
            connection: None, /* stream: None,
                               * header: None, */
        };
        println!("Starting to serve (Rust)");
        mw
    }

    // TODO: THIS NEEDS TO TAKE BUFFERS BACKED BY NUMPY ARRAYS
    /// Blocking call that waits for a new incoming connection
    pub fn wait_for_connection(&mut self,
                               recv_buffer_one: (*mut f64, usize),
                               send_buffer_one: (*mut f64, usize),
                               recv_buffer_two: (*mut f64, usize),
                               send_buffer_two: (*mut f64, usize)) {
        if self.connection.is_some() {
            panic!("Already connected to a client");
        } else {
            let (stream, _) = self.listener.accept().unwrap();
            stream.set_nonblocking(true).unwrap();
            // stream.set_nodelay(true).unwrap();
            let mut batch_vec = Vec::with_capacity(2);
            batch_vec.push(Batch::new(1,
                                      Buffer::new(recv_buffer_one),
                                      Buffer::new(send_buffer_one)));
            batch_vec.push(Batch::new(2,
                                      Buffer::new(recv_buffer_two),
                                      Buffer::new(send_buffer_two)));

            let mut batch_status = HashMap::new();
            for b in batch_vec.iter() {
                batch_status.insert(b.id, b.status.clone());
            }

            let (sender, receiver) = mpsc::channel::<(Header, u64, u64)>();
            let jh = {
                // let batches = batches.clone();
                thread::spawn(move || {
                    socket_event_loop_run(stream, batch_vec, sender);
                })
            };
            let mon_jh = {
                let rt_vec = self.read_times.clone();
                let qt_vec = self.queue_times.clone();
                thread::spawn(move || {
                    monitor_timing(rt_vec, qt_vec);
                })
            };
            self.connection = Some(Connection {
                batch_receiver: receiver,
                batch_status: batch_status,
                current_batch: None,
                event_loop_handle: jh,
            });
            println!("Handling new connection (Rust)");
        }
    }

    /// Block until a new batch is ready
    /// Returns the batch ID
    pub fn get_next_batch(&mut self) -> Header {
        let mut conn = self.connection.as_mut().unwrap();
        assert!(conn.current_batch.is_none());
        // blocking call to receive
        let (cur_batch_header, read_time, queue_start_time) = conn.batch_receiver.recv().unwrap();
        let queue_end_time = time::precise_time_ns();
        {
            let mut rt_lock = self.read_times.lock().unwrap();
            rt_lock.push(read_time);
        }
        {
            let mut qt_lock = self.queue_times.lock().unwrap();
            qt_lock.push(queue_end_time - queue_start_time);
        }
        conn.current_batch = Some(cur_batch_header.clone());
        cur_batch_header
    }

    fn finish_batch(&mut self, id: u32, num_responses: u32) {
        let mut conn = self.connection.as_mut().unwrap();
        assert!(conn.current_batch.is_some());
        assert!(conn.current_batch.as_ref().unwrap().batch_id == id);
        assert!(conn.current_batch.as_ref().unwrap().num_inputs == num_responses);
        let mut status_guard = conn.batch_status.get(&id).unwrap().lock().unwrap();
        *status_guard = BatchStatus::Write(num_responses * mem::size_of::<f64>() as u32);
        conn.current_batch = None;
    }
}

#[no_mangle]
pub extern "C" fn init_server(address: *const c_char) -> *mut BufferedServer {
    let addr_cstr = unsafe {
        assert!(!address.is_null());
        CStr::from_ptr(address)
    };
    let addr_str = str::from_utf8(addr_cstr.to_bytes()).unwrap();
    Box::into_raw(Box::new(BufferedServer::new(addr_str)))
}

#[no_mangle]
pub extern "C" fn server_free(ptr: *mut BufferedServer) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        Box::from_raw(ptr);
    }
}

#[no_mangle]
pub extern "C" fn wait_for_connection(ptr: *mut BufferedServer,
                                      recv_buffer_one: *mut f64,
                                      recv_buffer_one_len: u32,
                                      send_buffer_one: *mut f64,
                                      send_buffer_one_len: u32,
                                      recv_buffer_two: *mut f64,
                                      recv_buffer_two_len: u32,
                                      send_buffer_two: *mut f64,
                                      send_buffer_two_len: u32) {
    let mut server = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };
    server.wait_for_connection((recv_buffer_one, recv_buffer_one_len as usize),
                               (send_buffer_one, send_buffer_one_len as usize),
                               (recv_buffer_two, recv_buffer_two_len as usize),
                               (send_buffer_two, send_buffer_two_len as usize));
}

#[no_mangle]
pub extern "C" fn get_next_batch(ptr: *mut BufferedServer) -> Header {
    let mut server = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };
    server.get_next_batch()
}

#[no_mangle]
pub extern "C" fn finish_batch(ptr: *mut BufferedServer, id: u32, num_responses: u32) {
    let mut server = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };
    server.finish_batch(id, num_responses);
}
