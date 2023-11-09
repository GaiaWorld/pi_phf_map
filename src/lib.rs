#![feature(test)]
extern crate test;

/// 完美hash表，要求k一定不重复
/// 创建步骤：
/// 1、先根据总数量来初始化扩大3-6倍的容量，算法为：len*3 向上取2的幂。
/// 2、先用传入的哈希值，来计算所有k的hash，然后取高位右移落入容量范围，检查是否重复。
/// 3、如果重复，则将哈希值作为种子，随机取下一个哈希值，继续尝试。
/// 4、超过指定次数，默认为1000次，则扩容一倍，进行新的一轮尝试。
/// 5、直到找到一个hash值，能让所有的k都不重复的落在容量空间。然后做前后空间的裁剪来节省内存。
/// 测试get性能大概PhfMap为1ns, 标准的HashMap大概为3ns
/// 测试PhfMap::new的性能， 10个kv为200~400ns，30个kv为300~5000ns，100个kv为63~80us，500个kv大概为0.5~0.8ms，可以通过初始大小来大幅提高new的性能，如果想更快的创建，则可以考虑，将k分段，减少单表的数量到16-32左右。
/// 数组长度上， 10个kv一般为32，30个kv为128，100个kv为1024，500个kv大概为16384或32768
use core::fmt::*;
use core::hash::Hash;
use core::ops::{Index, IndexMut};
use fixedbitset::FixedBitSet;
use fxhash::FxHasher64;
use pi_null::Null;
use pi_wy_rng::WyRng;
use rand::{RngCore, SeedableRng};
use std::{hash::Hasher, marker::PhantomData, mem};

pub struct PhfMap<K: Hash, V: Null> {
    /// 值数组，空位为Null的V
    lut: Vec<V>,
    /// 元素的数量
    len: usize,
    /// hasher
    hasher: u64,
    /// 右移值
    right_shift: u32,
    /// 起始的偏移量
    offset: u32,
    _k: PhantomData<K>,
}

// 当前默认的HashMap和HashSet（使用根据平台字长、和feature来决定的DefaultHasher）
impl<K: Hash, V: Null> PhfMap<K, V> {
    /// 用指定的kv键创建完美hash表
    pub fn new(vec: Vec<(K, V)>) -> Self {
        Self::with_hasher(vec, 0)
    }
    /// 用指定的KV迭代器和指定的hasher创建完美hash表
    pub fn with_hasher(vec: Vec<(K, V)>, mut hasher: u64) -> Self {
        // 经验判断， 一般键数组长度的3～6倍的初始容量会比2～4倍的容量，在创建速度上快10倍
        // 扩大3～6倍
        let len = vec.len() as u64;
        let mut right_shift = (len * 3).leading_zeros();
        let size = 1 << (u64::BITS - right_shift);
        // println!("right_shift:{}, size:{:?}", right_shift, size);
        let mut conflicts = FixedBitSet::with_capacity(size);
        let mut count = 0;
        let mut r = WyRng::seed_from_u64(hasher);
        loop {
            if Self::test(&vec, hasher, right_shift, &mut conflicts) {
                // println!("count:{}, right_shift: {}, size:{:?}", count, right_shift, conflicts.len());
                return Self::make(vec, hasher, right_shift, conflicts);
            }
            count += 1;
            if count >= 1024 {
                // 扩容
                right_shift -= 1;
                let size = 1 << (u64::BITS - right_shift);
                conflicts = FixedBitSet::with_capacity(size);
                hasher = r.next_u64();
                count = 0;
                continue;
            }
            conflicts.clear();
            hasher = r.next_u64();
        }
    }
    fn test(vec: &Vec<(K, V)>, hasher: u64, right_shift: u32, conflicts: &mut FixedBitSet) -> bool {
        for (k, _v) in vec.iter() {
            let i = Self::hash(k, hasher, right_shift);
            if conflicts.contains(i) {
                return false;
            }
            conflicts.set(i, true);
        }
        true
    }
    fn make(vec: Vec<(K, V)>, hasher: u64, right_shift: u32, conflicts: FixedBitSet) -> Self {
        let mut offset = 0;
        for i in 0..conflicts.len() {
            if conflicts.contains(i) {
                break;
            }
            offset += 1;
        }
        let mut end = conflicts.len();
        for i in (0..conflicts.len()).rev() {
            if conflicts.contains(i) {
                break;
            }
            end -= 1;
        }
        let len = end - offset as usize;
        let mut lut = Vec::with_capacity(len);
        lut.extend((0..len).map(|_| V::null()));
        for (k, v) in vec.into_iter() {
            let i = Self::hash(&k, hasher, right_shift);
            lut[i - offset as usize] = v;
        }
        PhfMap {
            lut,
            len,
            hasher,
            right_shift,
            offset,
            _k: PhantomData,
        }
    }
    /// 获得kv表的容量
    pub fn len(&self) -> usize {
        self.len
    }
    /// 获得kv表的容量
    pub fn capacity(&self) -> usize {
        self.lut.len()
    }
    /// 获得指定键的只读引用
    #[inline(always)]
    pub fn get(&self, k: &K) -> Option<&V> {
        let h = self.location(k);
        self.lut.get(h)
    }
    /// 获得指定键的可写引用
    #[inline(always)]
    pub fn get_mut(&mut self, k: &K) -> Option<&mut V> {
        let h = self.location(k);
        self.lut.get_mut(h)
    }
    /// 获得指定键的只读引用
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, k: &K) -> &V {
        let h = self.location(k);
        self.lut.get_unchecked(h)
    }
    /// 获得指定键的可写引用
    #[inline(always)]
    pub unsafe fn get_unchecked_mut(&mut self, k: &K) -> &mut V {
        let h = self.location(k);
        self.lut.get_unchecked_mut(h)
    }
    #[inline(always)]
    pub fn val_iter(&self) -> Iter<'_, V> {
        Iter {
            lut: &self.lut,
            idx: 0,
            len: self.len,
        }
    }
    #[inline(always)]
    pub fn val_iter_mut(&mut self) -> IterMut<'_, V> {
        IterMut {
            lut: &mut self.lut,
            idx: 0,
            len: self.len,
        }
    }
    #[inline(always)]
    pub fn into_vec(self) -> Vec<V> {
        self.lut
    }
    /// 获得k的hash
    #[inline(always)]
    fn location(&self, k: &K) -> usize {
        Self::hash(k, self.hasher, self.right_shift).wrapping_sub(self.offset as usize)
    }
    /// 获得k的hash
    #[inline(always)]
    fn hash(k: &K, hasher: u64, right_shift: u32) -> usize {
        let mut state: FxHasher64 = unsafe { mem::transmute(hasher) };
        k.hash(&mut state);
        (state.finish() >> right_shift) as usize
    }
}
impl<K: Hash, V: Null> Index<K> for PhfMap<K, V> {
    type Output = V;
    fn index(&self, key: K) -> &V {
        self.get(&key).unwrap()
    }
}
impl<K: Hash, V: Null> IndexMut<K> for PhfMap<K, V> {
    fn index_mut(&mut self, key: K) -> &mut V {
        self.get_mut(&key).unwrap()
    }
}

impl<K: Hash, V: Null + Clone> Clone for PhfMap<K, V> {
    fn clone(&self) -> Self {
        PhfMap {
            lut: self.lut.clone(),
            len: self.len,
            hasher: self.hasher,
            right_shift: self.right_shift,
            offset: self.offset,
            _k: PhantomData,
        }
    }
}

impl<K: Hash, V: Null + Debug> Debug for PhfMap<K, V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let vec: Vec<&V> = self.val_iter().collect();
        f.debug_struct("PhfMap")
            .field("lut", &vec.as_slice())
            .field("len", &self.len)
            .field("hasher", &self.hasher)
            .field("right_shift", &self.right_shift)
            .field("offset", &self.offset)
            .finish()
    }
}

pub struct Iter<'a, T: Null> {
    lut: &'a Vec<T>,
    idx: usize,
    len: usize,
}

impl<'a, T: Null> Iterator for Iter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        while self.idx < self.lut.len() {
            let v = unsafe { self.lut.get_unchecked(self.idx) };
            self.idx += 1;
            if !v.is_null() {
                return Some(v);
            }
        }
        None
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}
impl<T: Null + Debug> Debug for Iter<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_tuple("Iter").field(&self.idx).finish()
    }
}

pub struct IterMut<'a, V: Null> {
    lut: &'a mut Vec<V>,
    idx: usize,
    len: usize,
}

impl<'a, V: Null> Iterator for IterMut<'a, V> {
    type Item = &'a mut V;
    fn next(&mut self) -> Option<Self::Item> {
        let len = self.lut.len();
        while self.idx < len {
            let v = unsafe { self.lut.get_unchecked_mut(self.idx) as *mut V };
            self.idx += 1;
            if !v.is_null() {
                return Some(unsafe { &mut *v });
            }
        }
        None
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}
#[cfg(test)]
mod test_mod {
    extern crate rand;

    use crate::*;
    use rand::Rng;
    use test::Bencher;

    #[test]
    fn test() {
        // 32 - 126
        // 128 - 600
        // 512 - 3200
        //let mut rng = rand::thread_rng();
        let mut rng = pi_wy_rng::WyRng::seed_from_u64(22222);
        let mut arr = Vec::new();
        for _ in 0..16 {
            let k = rng.gen::<usize>();
            arr.push((k, k + 1));
        }
        let map = PhfMap::with_hasher(arr.clone(), 0);
        println!("map:{:?}", map);
        for (k, v) in arr {
            let n = map[k];
            assert_eq!(n, v);
        }
    }

    #[bench]
    fn bench_make(b: &mut Bencher) {
        use rand::Rng;

        let mut rng = rand::thread_rng();
        let mut arr = Vec::new();
        for _ in 0..430 {
            let k = rng.gen::<usize>();
            arr.push((k, k + 1));
        }
        b.iter(move || {
            let map = PhfMap::with_hasher(arr.clone(), 0);
            for (k, v) in arr.iter() {
                let n = map.get(&k).unwrap();
                assert_eq!(n, v);
            }
        });
    }
    #[bench]
    fn bench_test(b: &mut Bencher) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut arr = Vec::new();
        for _ in 0..430 {
            let k = rng.gen::<usize>();
            arr.push((k, k + 1));
        }
        let map = PhfMap::with_hasher(arr.clone(), 0);
        println!("map capacity:{}", map.capacity());
        b.iter(move || {
            for &(k, v) in &arr {
                let n = map.get(&k).unwrap();
                assert_eq!(*n, v);
            }
        });
    }
    #[bench]
    fn bench_test_arr(b: &mut Bencher) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut arr = Vec::new();
        for _ in 0..430 {
            let k = rng.gen::<usize>();
            arr.push((k, k + 1));
        }
        let mut suffle = arr.clone();
        suffle.sort();
        println!("arr len:{}", 1);
        b.iter(move || {
            for &(k, _v) in &arr {
                let n = suffle.iter().position(|&x| x.0 == k);
                assert!(n.is_some());
            }
        });
    }
    #[bench]
    fn bench_test_map(b: &mut Bencher) {
        use rand::Rng;
        use std::collections::HashMap;
        use std::hash::BuildHasherDefault;

        pub type XHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher64>>;
        let mut rng = rand::thread_rng();
        let mut arr = Vec::new();
        let mut map: XHashMap<usize, usize> = XHashMap::with_hasher(Default::default());
        for _ in 0..430 {
            let k = rng.gen::<usize>();
            arr.push((k, k + 1));
            map.insert(k, k + 1);
        }
        b.iter(move || {
            for &(k, v) in &arr {
                let n = map.get(&k);
                assert_eq!(*n.unwrap(), v);
            }
        });
    }
}
