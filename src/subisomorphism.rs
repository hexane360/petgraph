use std::collections::HashSet;
use std::ops::Index;
use std::iter::repeat_with;
use std::hash::Hash;
use std::marker::PhantomData;

use crate::visit::{
	GraphBase, GraphRef, IntoNeighbors,
	NodeCompactIndexable, NodeIndexable,
	GraphProp, Data, //GetAdjacencyMatrix
};
use crate::Undirected;
use crate::data::DataMap;

use fixedbitset::FixedBitSet;

#[derive(Clone)]
struct FixedBitMatrix {
	m: Vec<FixedBitSet>
}

impl FixedBitMatrix {
	pub fn with_capacity(r: usize, c: usize) -> Self {
		Self {
			m: repeat_with(|| FixedBitSet::with_capacity(c))
				.take(r).collect()
		}
	}

	pub fn row(&self, r: usize) -> &FixedBitSet {
		assert!(self.m.len() >= r);
		&self.m[r]
	}

	pub fn row_mut(&mut self, r: usize) -> &mut FixedBitSet {
		assert!(self.m.len() >= r);
		&mut self.m[r]
	}

	pub fn set(&mut self, r: usize, c: usize, val: bool) {
		self.row_mut(r).set(c, val)
	}

	pub fn copy_row_from(&mut self, other: &Self, row: usize) {
		for (dest, &src) in self.m[row].as_mut_slice().iter_mut().zip(other.m[row].as_slice()) {
			*dest = src
		}
	}
}

pub struct SubgraphMapping<'a, G1, N1, N2> {
	graph: &'a G1,
	inner: Vec<N2>,
	marker: PhantomData<N1>,
}

impl<'a, G1, N1, N2> SubgraphMapping<'a, G1, N1, N2>
where &'a G1: GraphRef + GraphBase<NodeId = N1> + NodeIndexable + NodeCompactIndexable,
	  N1: Copy + PartialEq,
	  N2: Copy + PartialEq,
{
	/// Get the graph node matching the given pattern node in the isomorphism.
	/// O(1) complexity.
	pub fn get(&self, pattern_node: N1) -> N2 {
		self.inner[self.graph.to_index(pattern_node)]
	}

	fn get_index(&self, pattern_index: usize) -> N2 {
		self.inner[pattern_index]
	}

	/// Iterate through the node pairings in the isomorphism. The iterator is
	/// guaranteed to cover each pattern node, but not every graph node.
	pub fn iter<'b: 'a>(&'b self) -> impl Iterator<Item=(N1, N2)> + ExactSizeIterator + 'b {
		let graph: &'b G1 = &*self.graph;
		self.inner.iter().enumerate().map(move |(i, &n2)| (graph.from_index(i), n2))
	}

	/// Attempt to find the patttern node matching a given graph node.
	/// O(N(pattern)) complexity.
	pub fn search_graph_node(&self, graph_node: N2) -> Option<N1> {
		self.inner.iter().position(|&n| n == graph_node).map(|i| self.graph.from_index(i))
	}

	fn from_vec(graph: &'a G1, v: Vec<N2>) -> Self {
		Self {
			graph,
			inner: v,
			marker: PhantomData,
		}
	}
}

impl<'a, G1, N1, N2> Index<N1> for SubgraphMapping<'a, G1, N1, N2>
where &'a G1: GraphRef + GraphBase<NodeId = N1> + NodeIndexable + NodeCompactIndexable,
	  N1: Copy + PartialEq,
	  N2: Copy + PartialEq,
{
	type Output = N2;
	fn index(&self, pattern_node: N1) -> &N2 {
		&self.inner[self.graph.to_index(pattern_node)]
	}
}

/// Find a subgraph isomorphism - an injective mapping from `pattern` to `graph` which preserves adjacency.
/// Returns `None` if no isomorphism exists.
pub fn subgraph_isomorphism<'a, 'b, G1, G2, N1, N2>(pattern: &'a G1, graph: &'b G2) -> Option<SubgraphMapping<'a, G1, N1, N2>>
where &'a G1: GraphRef + GraphBase<NodeId = N1> + NodeIndexable + NodeCompactIndexable + GraphProp<EdgeType = Undirected> + IntoNeighbors,
&'b G2: GraphRef + GraphBase<NodeId = N2> + NodeIndexable + NodeCompactIndexable + GraphProp<EdgeType = Undirected> + IntoNeighbors,
	  N1: Copy + PartialEq + Eq + Hash,
	  N2: Copy + PartialEq + Eq + Hash,
{
	let rows = pattern.node_bound();
	let cols = graph.node_bound();

	if rows > cols {
		//graph isn't big enough to contain pattern
		return None;
	}

	// Matrix giving possible mappings from pattern nodes (rows) to graph nodes (cols).
	// Our goal is to find one node for each row such that no column is repeated
	let mut m = FixedBitMatrix::with_capacity(rows, cols);

	for row in 0..rows {
		for col in 0..cols {
			let possible = graph.neighbors(graph.from_index(col)).count() >=
				pattern.neighbors(pattern.from_index(row)).count();
			m.set(row, col, possible);
		}
	}

	run_subgraph_isomorphism(pattern, graph, m)
}

/// Find a subgraph isomorphism, additionally requiring node weights to match.
pub fn subgraph_isomorphism_matching<'a, 'b, G1, G2, N1, N2, NW1, NW2, F>(pattern: &'a G1, graph: &'b G2, node_match: F) -> Option<SubgraphMapping<'a, G1, N1, N2>>
where &'a G1: GraphRef + GraphBase<NodeId = N1> + NodeIndexable + NodeCompactIndexable + GraphProp<EdgeType = Undirected> + IntoNeighbors + Data<NodeWeight=NW1> + DataMap,
&'b G2: GraphRef + GraphBase<NodeId = N2> + NodeIndexable + NodeCompactIndexable + GraphProp<EdgeType = Undirected> + IntoNeighbors + Data<NodeWeight=NW2> + DataMap,
	  N1: Copy + PartialEq + Eq + Hash,
	  N2: Copy + PartialEq + Eq + Hash,
	  F: Fn(&NW1, &NW2) -> bool,
{
	let rows = pattern.node_bound();
	let cols = graph.node_bound();

	if rows > cols {
		//graph isn't big enough to contain pattern
		return None;
	}

	// Matrix giving possible mappings from pattern nodes (rows) to graph nodes (cols).
	// Our goal is to find one node for each row such that no column is repeated
	let mut m = FixedBitMatrix::with_capacity(rows, cols);

	for row in 0..rows {
		for col in 0..cols {
			let possible = node_match(pattern.node_weight(pattern.from_index(row)).unwrap(),
									  graph.node_weight(graph.from_index(col)).unwrap())
				&& graph.neighbors(graph.from_index(col)).count() >= pattern.neighbors(pattern.from_index(row)).count();
			m.set(row, col, possible);
		}
	}

	run_subgraph_isomorphism(pattern, graph, m)
}

fn run_subgraph_isomorphism<'a, 'b, G1, G2, N1, N2>(pattern: &'a G1, graph: &'b G2, m: FixedBitMatrix) -> Option<SubgraphMapping<'a, G1, N1, N2>>
where &'a G1: GraphRef + GraphBase<NodeId = N1> + NodeIndexable + NodeCompactIndexable + GraphProp<EdgeType = Undirected> + IntoNeighbors,
	  &'b G2: GraphRef + GraphBase<NodeId = N2> + NodeIndexable + NodeCompactIndexable + GraphProp<EdgeType = Undirected> + IntoNeighbors,
	  N1: Copy + PartialEq + Eq + Hash,
	  N2: Copy + PartialEq + Eq + Hash,
{
	let rows = pattern.node_bound();
	let cols = graph.node_bound();
	for row in 0..rows {
		if m.row(row).count_ones(..) == 0 {
			// no possibilities for this pattern node
			return None
		}
	}

	//enumerate through all possible isomorphisms given matrix
	let mut cols_used = FixedBitSet::with_capacity(cols); //records which columns have been tried
	let mut stack: Vec<usize> = Vec::with_capacity(rows); //stack to record which columns have been assigned
	let mut m_d = m.clone(); //working copy of matrix, refined for the current search
	let mut search_start = 0; //column to start search at
	// iterative depth first search through possible mappings
	loop { //breadth
		loop { //depth
			println!("Searching for column, start: {}, stack: {:?}", search_start, &stack);

			if m_d.row(stack.len()).count_ones(search_start..) == 0 {
				// no possibilities for this pattern node
				break
			}

			// find first possible column and set matrix accordingly
			// TODO optimize this using bitset methods
			match (search_start..cols).filter(|&c| {
				m_d.row(stack.len())[c] && !cols_used[c] //is possible and not already used
			}).next() {
				None => break,
				Some(c) => {
					println!("Selecting column {} at row {}", c, stack.len());
					//clear all but c in this row
					m_d.row_mut(stack.len()).clear();
					m_d.set(stack.len(), c, true);
					stack.push(c);
					cols_used.set(c, true);
					search_start = 0;
				}
			}

			if stack.len() == rows {
				println!("Checking for isomorphism, stack: {:?}", &stack);

				/*
				for col in (0..cols) {
					s.push_str(&format!(" {} ", graph[graph.from_index(col)]));
				}
				println!("{}", s);
				println!("{}+", repeat("---").take(cols).collect::<String>());
				s.clear();

				for row in (0..rows) {
					for col in (0..cols) {
						s.push_str(if m_d.row(row)[col] { " 1 " } else { " 0 " });
					}
					println!("{}| {}", s, pattern[pattern.from_index(row)]);
					s.clear();
				}
				*/

				let isomorphism = SubgraphMapping::from_vec(pattern, stack.iter().map(|&i| graph.from_index(i)).collect());

				let mut matches = true;
				for row in 0..rows {
					let graph_node = isomorphism.get_index(row);
					let graph_neighbors: HashSet<_> = graph.neighbors(graph_node).collect();
					if !pattern.neighbors(pattern.from_index(row)).all(|n| graph_neighbors.contains(&isomorphism[n])) {
						matches = false;
						break;
					}
				}
				if matches {
					return Some(isomorphism);
				} else {
					break
				}
			}
		}

		if stack.len() == 0 {
			return None;
		}

		// decrement depth and resume search elsewhere
		let c = stack.pop().unwrap();
		cols_used.set(c, false);
		//depth -= 1;
		//overwrite working matrix row with the original copy
		m_d.copy_row_from(&m, stack.len());
		//*m_d.row_mut(stack.len()) = m.row(stack.len()).clone();
		search_start = c+1;
	}
}

#[cfg(test)]
mod test {
	use crate::graph::{UnGraph, NodeIndex};
	use super::subgraph_isomorphism;

	#[test]
	fn test_subgraph_isomorphism() {
		let mut pattern = UnGraph::<&str, ()>::new_undirected();
		pattern.add_node("0");
		pattern.add_node("1");
		pattern.add_node("2");
		pattern.add_node("3");
		pattern.extend_with_edges(&[(0, 1), (1, 2), (2, 0), (0, 3)]);

		let mut graph = UnGraph::<&str, ()>::new_undirected();
		graph.add_node("a");
		graph.add_node("b");
		graph.add_node("c");
		graph.add_node("d");
		graph.add_node("e");
		graph.extend_with_edges(&[(1, 2), (2, 3), (3, 1), (1, 4), (3, 0), (1, 0)]);

		let isomorphism = subgraph_isomorphism(&pattern, &graph);
		assert!(isomorphism.is_some());
		let isomorphism = isomorphism.unwrap();

		//assert!(isomorphism.get(&0).is_some());
		//assert!(isomorphism.get(&1).is_some());
		//assert!(isomorphism.get(&2).is_some());
		//assert!(isomorphism.get(&3).is_some());
		let ids: Vec<_> = (0..4).map(|i| isomorphism.get_index(i)).collect();

		//check that the required adjacencies are present
		assert!(graph.neighbors(ids[0]).count() >= 3);
		assert!(graph.neighbors(ids[0]).any(|e| e == ids[1]));
		assert!(graph.neighbors(ids[0]).any(|e| e == ids[2]));
		assert!(graph.neighbors(ids[0]).any(|e| e == ids[3]));
		assert!(graph.neighbors(ids[1]).count() >= 1);
		assert!(graph.neighbors(ids[1]).any(|e| e == ids[2]));
	}
}
