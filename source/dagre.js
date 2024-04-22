
const dagre = {};

// Dagre graph layout
// https://github.com/dagrejs/dagre
// https://github.com/dagrejs/graphlib

dagre.layout = (nodes, edges, layout, state) => {

    let uniqueIdCounter = 0;
    const uniqueId = (prefix) => {
        const id = ++uniqueIdCounter;
        return prefix + id;
    };

    const flat = (list) => {
        if (Array.isArray(list) && list.every((item) => !Array.isArray(item))) {
            return list;
        }
        const target = [];
        for (const item of list) {
            if (!Array.isArray(item)) {
                target.push(item);
                continue;
            }
            for (const entry of item) {
                target.push(entry);
            }
        }
        return target;
    };

    // Adds a dummy node to the graph and return v.
    const addDummyNode = (g, type, label, name) => {
        let v = '';
        do {
            v = uniqueId(name);
        } while (g.hasNode(v));
        label.dummy = type;
        g.setNode(v, label);
        return v;
    };

    const asNonCompoundGraph = (g) => {
        const graph = new dagre.Graph(true, false);
        for (const node of g.nodes.values()) {
            const v = node.v;
            if (g.children(v).length === 0) {
                graph.setNode(v, node.label);
            }
        }
        for (const e of g.edges.values()) {
            graph.setEdge(e.v, e.w, e.label);
        }
        return graph;
    };

    const maxRank = (g) => {
        let rank = Number.NEGATIVE_INFINITY;
        for (const node of g.nodes.values()) {
            const x = node.label.rank;
            if (x !== undefined && x > rank) {
                rank = x;
            }
        }
        return rank === Number.NEGATIVE_INFINITY ? undefined : rank;
    };

    // Given a DAG with each node assigned 'rank' and 'order' properties, this function will produce a matrix with the ids of each node.
    const buildLayerMatrix = (g) => {
        const rank = maxRank(g);
        const length = rank === undefined ? 0 : rank + 1;
        const layering = Array.from(new Array(length), () => []);
        for (const node of g.nodes.values()) {
            const label = node.label;
            const rank = label.rank;
            if (rank !== undefined) {
                layering[rank][label.order] = node.v;
            }
        }
        return layering;
    };

    // This idea comes from the Gansner paper: to account for edge labels in our layout we split each rank in half by doubling minlen and halving ranksep.
    // Then we can place labels at these mid-points between nodes.
    // We also add some minimal padding to the width to push the label for the edge away from the edge itself a bit.
    const makeSpaceForEdgeLabels = (g, state, layout) => {
        layout.ranksep /= 2;
        const rankdir = layout.rankdir;
        for (const e of g.edges.values()) {
            const edge = e.label;
            edge.minlen *= 2;
            if (edge.labelpos.toLowerCase() !== 'c') {
                if (rankdir === 'TB' || rankdir === 'BT') {
                    edge.width += edge.labeloffset;
                } else {
                    edge.height += edge.labeloffset;
                }
            }
        }
    };

    const removeSelfEdges = (g) => {
        for (const e of g.edges.values()) {
            if (e.v === e.w) {
                const label = e.vNode.label;
                if (!label.selfEdges) {
                    label.selfEdges = [];
                }
                label.selfEdges.push({ e, label: e.label });
                g.removeEdge(e);
            }
        }
    };

    const acyclic_run = (g) => {
        const edges = [];
        const visited = new Set();
        const path = new Set();
        const stack = Array.from(g.nodes.keys()).reverse();
        while (stack.length > 0) {
            const v = stack.pop();
            if (Array.isArray(v)) {
                path.delete(v[0]);
            } else if (!visited.has(v)) {
                visited.add(v);
                path.add(v);
                stack.push([v]);
                const out = g.node(v).out;
                for (let i = out.length - 1; i >= 0; i--) {
                    const e = out[i];
                    if (path.has(e.w)) {
                        edges.push(e);
                    }
                    stack.push(e.w);
                }
            }
        }
        for (const e of edges) {
            const label = e.label;
            g.removeEdge(e);
            label.forwardName = e.name;
            label.reversed = true;
            g.setEdge(e.w, e.v, label, uniqueId('rev'));
        }
    };
    const acyclic_undo = (g) => {
        for (const e of g.edges.values()) {
            const edge = e.label;
            if (edge.reversed) {
                edge.points.reverse();
                g.removeEdge(e);
                const forwardName = edge.forwardName;
                delete edge.reversed;
                delete edge.forwardName;
                g.setEdge(e.w, e.v, edge, forwardName);
            }
        }
    };

    // Returns the amount of slack for the given edge.
    // The slack is defined as the difference between the length of the edge and its minimum length.
    const slack = (g, e) => {
        return e.wNode.label.rank - e.vNode.label.rank - e.label.minlen;
    };

    // Assigns a rank to each node in the input graph that respects the 'minlen' constraint specified on edges between nodes.
    // This basic structure is derived from Gansner, et al., 'A Technique for Drawing Directed Graphs.'
    //
    // Pre-conditions:
    //    1. Graph must be a connected DAG
    //    2. Graph nodes must be objects
    //    3. Graph edges must have 'weight' and 'minlen' attributes
    //
    // Post-conditions:
    //    1. Graph nodes will have a 'rank' attribute based on the results of the
    //       algorithm. Ranks can start at any index (including negative), we'll
    //       fix them up later.
    const rank = (g) => {
        g = asNonCompoundGraph(g);
        // Constructs a spanning tree with tight edges and adjusted the input node's ranks to achieve this.
        // A tight edge is one that is has a length that matches its 'minlen' attribute.
        // The basic structure for this function is derived from Gansner, et al., 'A Technique for Drawing Directed Graphs.'
        //
        // Pre-conditions:
        //    1. Graph must be a DAG.
        //    2. Graph must be connected.
        //    3. Graph must have at least one node.
        //    5. Graph nodes must have been previously assigned a 'rank' property that respects the 'minlen' property of incident edges.
        //    6. Graph edges must have a 'minlen' property.
        //
        // Post-conditions:
        //    - Graph nodes will have their rank adjusted to ensure that all edges are tight.
        //
        // Returns a tree (undirected graph) that is constructed using only 'tight' edges.
        const feasibleTree = (g) => {
            const t = new dagre.Graph(false, false);
            // Choose arbitrary node from which to start our tree
            const start = g.nodes.keys().next().value;
            const size = g.nodes.size;
            t.setNode(start, {});
            // Finds a maximal tree of tight edges and returns the number of nodes in the tree.
            const tightTree = (t, g) => {
                const stack = Array.from(t.nodes.keys()).reverse();
                while (stack.length > 0) {
                    const v = stack.pop();
                    const node = g.node(v);
                    for (const e of node.in.concat(node.out)) {
                        const edgeV = e.v;
                        const w = (v === edgeV) ? e.w : edgeV;
                        if (!t.hasNode(w) && !slack(g, e)) {
                            t.setNode(w, {});
                            t.setEdge(v, w, {});
                            stack.push(w);
                        }
                    }
                }
                return t.nodes.size;
            };
            while (tightTree(t, g) < size) {
                // Finds the edge with the smallest slack that is incident on tree and returns it.
                let minKey = Number.MAX_SAFE_INTEGER;
                let edge = null;
                for (const e of g.edges.values()) {
                    if (t.hasNode(e.v) !== t.hasNode(e.w)) {
                        const key = slack(g, e);
                        if (key < minKey) {
                            minKey = key;
                            edge = e;
                        }
                    }
                }
                const delta = t.hasNode(edge.v) ? slack(g, edge) : -slack(g, edge);
                for (const v of t.nodes.keys()) {
                    g.node(v).label.rank += delta;
                }
            }
            return t;
        };
        // Initializes ranks for the input graph using the longest path algorithm.
        // This algorithm scales well and is fast in practice, it yields rather poor solutions.
        // Nodes are pushed to the lowest layer possible, leaving the bottom ranks wide and leaving edges longer than necessary.
        // However, due to its speed, this algorithm is good for getting an initial ranking that can be fed into other algorithms.
        //
        // This algorithm does not normalize layers because it will be used by other algorithms in most cases.
        // If using this algorithm directly, be sure to run normalize at the end.
        //
        // Pre-conditions:
        //    1. Input graph is a DAG.
        //    2. Input graph node labels can be assigned properties.
        //
        // Post-conditions:
        //    1. Each node will be assign an (unnormalized) 'rank' property.
        const longestPath = (g) => {
            const visited = new Set();
            const stack = [Array.from(g.nodes.values()).filter((node) => node.in.length === 0).reverse()];
            while (stack.length > 0) {
                const current = stack[stack.length - 1];
                if (Array.isArray(current)) {
                    const node = current.pop();
                    if (current.length === 0) {
                        stack.pop();
                    }
                    if (!visited.has(node)) {
                        visited.add(node);
                        const children = node.out.map((e) => e.wNode);
                        if (children.length > 0) {
                            stack.push(node);
                            stack.push(children.reverse());
                        } else {
                            node.label.rank = 0;
                        }
                    }
                } else {
                    stack.pop();
                    let rank = Number.MAX_SAFE_INTEGER;
                    for (const e of current.out) {
                        rank = Math.min(rank, e.wNode.label.rank - e.label.minlen);
                    }
                    current.label.rank = rank;
                }
            }
        };
        // The network simplex algorithm assigns ranks to each node in the input graph
        // and iteratively improves the ranking to reduce the length of edges.
        //
        // Preconditions:
        //    1. The input graph must be a DAG.
        //    2. All nodes in the graph must have an object value.
        //    3. All edges in the graph must have 'minlen' and 'weight' attributes.
        //
        // Postconditions:
        //    1. All nodes in the graph will have an assigned 'rank' attribute that has
        //       been optimized by the network simplex algorithm. Ranks start at 0.
        //
        // A rough sketch of the algorithm is as follows:
        //    1. Assign initial ranks to each node. We use the longest path algorithm,
        //       which assigns ranks to the lowest position possible. In general this
        //       leads to very wide bottom ranks and unnecessarily long edges.
        //    2. Construct a feasible tight tree. A tight tree is one such that all
        //       edges in the tree have no slack (difference between length of edge
        //       and minlen for the edge). This by itself greatly improves the assigned
        //       rankings by shorting edges.
        //    3. Iteratively find edges that have negative cut values. Generally a
        //       negative cut value indicates that the edge could be removed and a new
        //       tree edge could be added to produce a more compact graph.
        //
        // Much of the algorithms here are derived from Gansner, et al., 'A Technique
        // for Drawing Directed Graphs.' The structure of the file roughly follows the
        // structure of the overall algorithm.
        const networkSimplex = (g) => {
            // Returns a new graph with only simple edges. Handles aggregation of data associated with multi-edges.
            const simplify = (g) => {
                const graph = new dagre.Graph(true, false);
                for (const node of g.nodes.values()) {
                    graph.setNode(node.v, node.label);
                }
                for (const e of g.edges.values()) {
                    const simpleEdge =  graph.edge(e.v, e.w);
                    const simpleLabel = simpleEdge ? simpleEdge.label : { weight: 0, minlen: 1 };
                    const label = e.label;
                    graph.setEdge(e.v, e.w, {
                        weight: simpleLabel.weight + label.weight,
                        minlen: Math.max(simpleLabel.minlen, label.minlen)
                    });
                }
                return graph;
            };
            const initLowLimValues = (tree, root) => {
                const dfs = (tree, visited, nextLim, v, parent) => {
                    const low = nextLim;
                    const label = tree.node(v).label;
                    visited.add(v);
                    for (const w of tree.neighbors(v)) {
                        if (!visited.has(w)) {
                            nextLim = dfs(tree, visited, nextLim, w, v);
                        }
                    }
                    label.low = low;
                    label.lim = nextLim++;
                    if (parent) {
                        label.parent = parent;
                    } else {
                        // should be able to remove this when we incrementally update low lim
                        delete label.parent;
                    }
                    return nextLim;
                };
                root = tree.nodes.keys().next().value;
                const visited = new Set();
                dfs(tree, visited, 1, root);
            };
            // Initializes cut values for all edges in the tree.
            const initCutValues = (t, g) => {
                const vs = [];
                const visited = new Set();
                const stack = [Array.from(t.nodes.keys()).reverse()];
                while (stack.length > 0) {
                    const current = stack[stack.length - 1];
                    if (Array.isArray(current)) {
                        const v = current.pop();
                        if (current.length === 0) {
                            stack.pop();
                        }
                        if (!visited.has(v)) {
                            visited.add(v);
                            const children = t.neighbors(v);
                            if (children.length > 0) {
                                stack.push(v);
                                stack.push(children.reverse());
                            } else {
                                vs.push(v);
                            }
                        }
                    } else {
                        vs.push(stack.pop());
                    }
                }
                for (const v of vs.slice(0, vs.length - 1)) {
                    // Given the tight tree, its graph, and a child in the graph calculate and
                    // return the cut value for the edge between the child and its parent.
                    const childLabel = t.node(v).label;
                    const parent = childLabel.parent;
                    // The graph's view of the tree edge we're inspecting
                    const edge = g.edge(v, parent);
                    // True if the child is on the tail end of the edge in the directed graph
                    const childIsTail = edge ? true : false;
                    // The accumulated cut value for the edge between this node and its parent
                    const graphEdge = edge ? edge.label : g.edge(parent, v).label;
                    let cutValue = graphEdge.weight;
                    const node = g.node(v);
                    for (const e of node.in.concat(node.out)) {
                        const isOutEdge = e.v === v;
                        const other = isOutEdge ? e.w : e.v;
                        if (other !== parent) {
                            const pointsToHead = isOutEdge === childIsTail;
                            cutValue += pointsToHead ? e.label.weight : -e.label.weight;
                            const edge = t.edge(v, other);
                            if (edge) {
                                const otherCutValue = edge.label.cutvalue;
                                cutValue += pointsToHead ? -otherCutValue : otherCutValue;
                            }
                        }
                    }
                    t.edge(v, parent).label.cutvalue = cutValue;
                }
            };
            const leaveEdge = (tree) => {
                return Array.from(tree.edges.values()).find((e) => e.label.cutvalue < 0);
            };
            const enterEdge = (t, g, edge) => {
                let v = edge.v;
                let w = edge.w;
                // For the rest of this function we assume that v is the tail and w is the
                // head, so if we don't have this edge in the graph we should flip it to
                // match the correct orientation.
                if (!g.edge(v, w)) {
                    v = edge.w;
                    w = edge.v;
                }
                const vLabel = t.node(v).label;
                const wLabel = t.node(w).label;
                let tailLabel = vLabel;
                let flip = false;
                // If the root is in the tail of the edge then we need to flip the logic that
                // checks for the head and tail nodes in the candidates function below.
                if (vLabel.lim > wLabel.lim) {
                    tailLabel = wLabel;
                    flip = true;
                }
                // Returns true if the specified node is descendant of the root node per the assigned low and lim attributes in the tree.
                const isDescendant = (vLabel, rootLabel) => {
                    return rootLabel.low <= vLabel.lim && vLabel.lim <= rootLabel.lim;
                };
                let minKey = Number.POSITIVE_INFINITY;
                let minValue = null;
                for (const edge of g.edges.values()) {
                    if (flip === isDescendant(t.node(edge.v).label, tailLabel) &&
                        flip !== isDescendant(t.node(edge.w).label, tailLabel)) {
                        const key = slack(g, edge);
                        if (key < minKey) {
                            minKey = key;
                            minValue = edge;
                        }
                    }
                }
                return minValue;
            };
            const exchangeEdges = (t, g, e, f) => {
                t.removeEdge(e);
                t.setEdge(f.v, f.w, {});
                initLowLimValues(t);
                initCutValues(t, g);
                // update ranks
                const root = Array.from(t.nodes.keys()).find((v) => !g.node(v).label.parent);
                const stack = [root];
                const visited = new Set();
                while (stack.length > 0) {
                    const v = stack.pop();
                    if (!visited.has(v)) {
                        visited.add(v);
                        const neighbors = t.neighbors(v);
                        for (let i = neighbors.length - 1; i >= 0; i--) {
                            stack.push(neighbors[i]);
                        }
                    }
                }
                const vs = Array.from(visited);
                for (const v of vs.slice(1)) {
                    const parent = t.node(v).label.parent;
                    let edge = g.edge(v, parent);
                    let flipped = false;
                    if (!edge) {
                        edge = g.edge(parent, v);
                        flipped = true;
                    }
                    g.node(v).label.rank = g.node(parent).label.rank + (flipped ? edge.label.minlen : -edge.label.minlen);
                }
            };
            g = simplify(g);
            longestPath(g);
            const t = feasibleTree(g);
            initLowLimValues(t);
            initCutValues(t, g);
            let e = null;
            let f = null;
            while ((e = leaveEdge(t))) {
                f = enterEdge(t, g, e);
                exchangeEdges(t, g, e, f);
            }
        };
        switch (layout.ranker) {
            case 'tight-tree':
                longestPath(g);
                feasibleTree(g);
                break;
            case 'longest-path':
                longestPath(g);
                break;
            default:
                networkSimplex(g);
                break;
        }
    };

    // Creates temporary dummy nodes that capture the rank in which each edge's label is going to, if it has one of non-zero width and height.
    // We do this so that we can safely remove empty ranks while preserving balance for the label's position.
    const injectEdgeLabelProxies = (g) => {
        for (const e of g.edges.values()) {
            const edge = e.label;
            if (edge.width && edge.height) {
                const v = e.vNode.label;
                const w = e.wNode.label;
                addDummyNode(g, 'edge-proxy', { rank: (w.rank - v.rank) / 2 + v.rank, e }, '_ep');
            }
        }
    };

    const removeEmptyRanks = (g, state) => {
        // Ranks may not start at 0, so we need to offset them
        if (g.nodes.size > 0) {
            let minRank = Number.MAX_SAFE_INTEGER;
            let maxRank = Number.MIN_SAFE_INTEGER;
            const nodes = Array.from(g.nodes.values());
            for (const node of nodes) {
                const label = node.label;
                if (label.rank !== undefined) {
                    minRank = Math.min(minRank, label.rank);
                    maxRank = Math.max(maxRank, label.rank);
                }
            }
            const size = maxRank - minRank;
            if (size > 0) {
                const layers = new Array(size);
                for (const node of nodes) {
                    const label = node.label;
                    if (label.rank !== undefined) {
                        const rank = label.rank - minRank;
                        if (!layers[rank]) {
                            layers[rank] = [];
                        }
                        layers[rank].push(node.v);
                    }
                }
                let delta = 0;
                const nodeRankFactor = state.nodeRankFactor;
                for (let i = 0; i < layers.length; i++) {
                    const vs = layers[i];
                    if (vs === undefined && i % nodeRankFactor !== 0) {
                        delta--;
                    } else if (delta && vs) {
                        for (const v of vs) {
                            g.node(v).label.rank += delta;
                        }
                    }
                }
            }
        }
    };

    // A nesting graph creates dummy nodes for the tops and bottoms of subgraphs,
    // adds appropriate edges to ensure that all cluster nodes are placed between
    // these boundries, and ensures that the graph is connected.
    // In addition we ensure, through the use of the minlen property, that nodes
    // and subgraph border nodes do not end up on the same rank.
    //
    // Preconditions:
    //    1. Input graph is a DAG
    //    2. Nodes in the input graph has a minlen attribute
    //
    // Postconditions:
    //   1. Input graph is connected.
    //   2. Dummy nodes are added for the tops and bottoms of subgraphs.
    //   3. The minlen attribute for nodes is adjusted to ensure nodes do not
    //      get placed on the same rank as subgraph border nodes.
    //
    // The nesting graph idea comes from Sander, 'Layout of Compound Directed Graphs.'
    const nestingGraph_run = (g, state) => {
        const root = addDummyNode(g, 'root', {}, '_root');
        const treeDepths = (g) => {
            const depths = {};
            const dfs = (v, depth) => {
                const children = g.children(v);
                if (children && children.length > 0) {
                    for (const child of children) {
                        dfs(child, depth + 1);
                    }
                }
                depths[v] = depth;
            };
            for (const v of g.children()) {
                dfs(v, 1);
            }
            return depths;
        };
        const dfs = (g, root, nodeSep, weight, height, depths, v) => {
            const children = g.children(v);
            if (!children.length) {
                if (v !== root) {
                    g.setEdge(root, v, { weight: 0, minlen: nodeSep });
                }
                return;
            }
            const top = addDummyNode(g, 'border', { width: 0, height: 0 }, '_bt');
            const bottom = addDummyNode(g, 'border', { width: 0, height: 0 }, '_bb');
            const label = g.node(v).label;
            g.setParent(top, v);
            label.borderTop = top;
            g.setParent(bottom, v);
            label.borderBottom = bottom;
            for (const child of children) {
                dfs(g, root, nodeSep, weight, height, depths, child);
                const childNode = g.node(child).label;
                const childTop = childNode.borderTop ? childNode.borderTop : child;
                const childBottom = childNode.borderBottom ? childNode.borderBottom : child;
                const thisWeight = childNode.borderTop ? weight : 2 * weight;
                const minlen = childTop === childBottom ? height - depths[v] + 1 : 1;
                g.setEdge(top, childTop, { weight: thisWeight, minlen, nestingEdge: true });
                g.setEdge(childBottom, bottom, { weight: thisWeight, minlen, nestingEdge: true });
            }
            if (!g.parent(v)) {
                g.setEdge(root, top, { weight: 0, minlen: height + depths[v] });
            }
        };
        const depths = treeDepths(g);
        const height = Math.max(...Object.values(depths)) - 1; // Note: depths is an Object not an array
        const nodeSep = 2 * height + 1;
        state.nestingRoot = root;
        // Multiply minlen by nodeSep to align nodes on non-border ranks.
        for (const e of g.edges.values()) {
            e.label.minlen *= nodeSep;
        }
        // Calculate a weight that is sufficient to keep subgraphs vertically compact
        const weight = Array.from(g.edges.values()).reduce((acc, e) => acc + e.label.weight, 0) + 1;
        // Create border nodes and link them up
        for (const child of g.children()) {
            dfs(g, root, nodeSep, weight, height, depths, child);
        }
        // Save the multiplier for node layers for later removal of empty border layers.
        state.nodeRankFactor = nodeSep;
    };

    const nestingGraph_cleanup = (g, state) => {
        g.removeNode(state.nestingRoot);
        delete state.nestingRoot;
        for (const e of g.edges.values()) {
            if (e.label.nestingEdge) {
                g.removeEdge(e);
            }
        }
    };

    const assignRankMinMax = (g, state) => {
        // Adjusts the ranks for all nodes in the graph such that all nodes v have rank(v) >= 0 and at least one node w has rank(w) = 0.
        let min = Number.POSITIVE_INFINITY;
        for (const node of g.nodes.values()) {
            const rank = node.label.rank;
            if (rank !== undefined && rank < min) {
                min = rank;
            }
        }
        for (const node of g.nodes.values()) {
            const label = node.label;
            if (label.rank !== undefined) {
                label.rank -= min;
            }
        }
        let maxRank = 0;
        for (const node of g.nodes.values()) {
            const label = node.label;
            if (label.borderTop) {
                label.minRank = g.node(label.borderTop).label.rank;
                label.maxRank = g.node(label.borderBottom).label.rank;
                maxRank = Math.max(maxRank, label.maxRank);
            }
        }
        state.maxRank = maxRank;
    };

    // Breaks any long edges in the graph into short segments that span 1 layer each.
    // This operation is undoable with the denormalize function.
    //
    // Pre-conditions:
    //   1. The input graph is a DAG.
    //   2. Each node in the graph has a 'rank' property.
    //
    // Post-condition:
    //   1. All edges in the graph have a length of 1.
    //   2. Dummy nodes are added where edges have been split into segments.
    //   3. The graph is augmented with a 'dummyChains' attribute which contains
    //      the first dummy in each chain of dummy nodes produced.
    const normalize = (g, state) => {
        state.dummyChains = [];
        for (const e of g.edges.values()) {
            let v = e.v;
            const w = e.w;
            const name = e.name;
            const edgeLabel = e.label;
            const labelRank = edgeLabel.labelRank;
            let vRank = g.node(v).label.rank;
            const wRank = g.node(w).label.rank;
            if (wRank !== vRank + 1) {
                g.removeEdge(e);
                let first = true;
                vRank++;
                while (vRank < wRank) {
                    edgeLabel.points = [];
                    delete e.key;
                    const attrs = {
                        width: 0, height: 0,
                        edgeLabel,
                        edgeObj: e,
                        rank: vRank
                    };
                    const dummy = addDummyNode(g, 'edge', attrs, '_d');
                    if (vRank === labelRank) {
                        attrs.width = edgeLabel.width;
                        attrs.height = edgeLabel.height;
                        attrs.dummy = 'edge-label';
                        attrs.labelpos = edgeLabel.labelpos;
                    }
                    g.setEdge(v, dummy, { weight: edgeLabel.weight }, name);
                    if (first) {
                        state.dummyChains.push(dummy);
                        first = false;
                    }
                    v = dummy;
                    vRank++;
                }
                g.setEdge(v, w, { weight: edgeLabel.weight }, name);
            }
        }
    };

    const denormalize = (g, state) => {
        for (let v of state.dummyChains) {
            let label = g.node(v).label;
            const edgeLabel = label.edgeLabel;
            const e = label.edgeObj;
            g.setEdge(e.v, e.w, edgeLabel, e.name);
            while (label.dummy) {
                const [w] = g.successors(v);
                g.removeNode(v);
                edgeLabel.points.push({ x: label.x, y: label.y });
                if (label.dummy === 'edge-label') {
                    edgeLabel.x = label.x;
                    edgeLabel.y = label.y;
                    edgeLabel.width = label.width;
                    edgeLabel.height = label.height;
                }
                v = w;
                label = g.node(v).label;
            }
        }
    };

    const removeEdgeLabelProxies = (g) => {
        for (const node of g.nodes.values()) {
            const label = node.label;
            if (label.dummy === 'edge-proxy') {
                label.e.label.labelRank = label.rank;
                g.removeNode(node.v);
            }
        }
    };

    const parentDummyChains = (g, state) => {
        // Find a path from v to w through the lowest common ancestor (LCA). Return the full path and the LCA.
        const findPath = (g, postorderNums, v, w) => {
            const low = Math.min(postorderNums[v].low, postorderNums[w].low);
            const lim = Math.max(postorderNums[v].lim, postorderNums[w].lim);
            // Traverse up from v to find the LCA
            let parent = v;
            const vPath = [];
            do {
                parent = g.parent(parent);
                vPath.push(parent);
            }
            while (parent && (postorderNums[parent].low > low || lim > postorderNums[parent].lim));
            const lca = parent;
            // Traverse from w to LCA
            parent = w;
            const wPath = [];
            while ((parent = g.parent(parent)) !== lca) {
                wPath.push(parent);
            }
            return { path: vPath.concat(wPath.reverse()), lca };
        };
        const postorder = (g) => {
            const result = {};
            let lim = 0;
            const dfs = (v) => {
                const low = lim;
                for (const u of g.children(v)) {
                    dfs(u);
                }
                result[v] = { low, lim: lim++ };
            };
            for (const v of g.children()) {
                dfs(v);
            }
            return result;
        };
        const postorderNums = postorder(g);
        for (let v of state.dummyChains || []) {
            const node = g.node(v).label;
            const edgeObj = node.edgeObj;
            const pathData = findPath(g, postorderNums, edgeObj.v, edgeObj.w);
            const path = pathData.path;
            const lca = pathData.lca;
            let pathIdx = 0;
            let pathV = path[pathIdx];
            let ascending = true;
            while (v !== edgeObj.w) {
                const node = g.node(v).label;
                if (ascending) {
                    while ((pathV = path[pathIdx]) !== lca && g.node(pathV).label.maxRank < node.rank) {
                        pathIdx++;
                    }
                    if (pathV === lca) {
                        ascending = false;
                    }
                }
                if (!ascending) {
                    while (pathIdx < path.length - 1 && g.node(path[pathIdx + 1]).label.minRank <= node.rank) {
                        pathIdx++;
                    }
                    pathV = path[pathIdx];
                }
                g.setParent(v, pathV);
                [v] = g.successors(v);
            }
        }
    };

    const addBorderSegments = (g) => {
        const addBorderNode = (g, prop, prefix, sg, sgNode, rank) => {
            const label = { width: 0, height: 0, rank, borderType: prop };
            const prev = sgNode[prop][rank - 1];
            const curr = addDummyNode(g, 'border', label, prefix);
            sgNode[prop][rank] = curr;
            g.setParent(curr, sg);
            if (prev) {
                g.setEdge(prev, curr, { weight: 1 });
            }
        };
        const queue = g.children();
        for (let i = 0; i < queue.length; i++) {
            const v = queue[i];
            const node = g.node(v).label;
            if ('minRank' in node) {
                node.borderLeft = [];
                node.borderRight = [];
                const maxRank = node.maxRank + 1;
                for (let rank = node.minRank; rank < maxRank; rank++) {
                    addBorderNode(g, 'borderLeft', '_bl', v, node, rank);
                    addBorderNode(g, 'borderRight', '_br', v, node, rank);
                }
            }
            const children = g.children(v);
            if (children.length) {
                for (const v of children) {
                    queue.push(v);
                }
            }
        }
    };

    // Applies heuristics to minimize edge crossings in the graph and sets the best order solution as an order attribute on each node.
    //
    // Pre-conditions:
    //    1. Graph must be DAG
    //    2. Graph nodes must have the 'rank' attribute
    //    3. Graph edges must have the 'weight' attribute
    //
    // Post-conditions:
    //    1. Graph nodes will have an 'order' attribute based on the results of the algorithm.
    const order = (g) => {
        const sortSubgraph = (g, v, cg, biasRight) => {
            // Given a list of entries of the form {v, barycenter, weight} and a constraint graph this function will resolve any conflicts between the constraint graph and the barycenters for the entries.
            // If the barycenters for an entry would violate a constraint in the constraint graph then we coalesce the nodes in the conflict into a new node that respects the contraint and aggregates barycenter and weight information.
            // This implementation is based on the description in Forster, 'A Fast and Simple Hueristic for Constrained Two-Level Crossing Reduction,' thought it differs in some specific details.
            //
            // Pre-conditions:
            //    1. Each entry has the form {v, barycenter, weight}, or if the node has no barycenter, then {v}.
            //
            // Returns:
            //    A new list of entries of the form {vs, i, barycenter, weight}.
            //    The list `vs` may either be a singleton or it may be an aggregation of nodes ordered such that they do not violate constraints from the constraint graph.
            //    The property `i` is the lowest original index of any of the elements in `vs`.
            const resolveConflicts = (entries, cg) => {
                const mappedEntries = new Map();
                for (let i = 0; i < entries.length; i++) {
                    const entry = entries[i];
                    const tmp = { indegree: 0, 'in': [], out: [], vs: [entry.v], i };
                    if (entry.barycenter !== undefined) {
                        tmp.barycenter = entry.barycenter;
                        tmp.weight = entry.weight;
                    }
                    mappedEntries.set(entry.v, tmp);
                }
                for (const e of cg.edges.values()) {
                    const entryV = mappedEntries.get(e.v);
                    const entryW = mappedEntries.get(e.w);
                    if (entryV && entryW) {
                        entryW.indegree++;
                        entryV.out.push(entryW);
                    }
                }
                const sourceSet = Array.from(mappedEntries.values()).filter((entry) => !entry.indegree);
                const results = [];
                function handleIn(vEntry) {
                    return function(uEntry) {
                        if (uEntry.merged) {
                            return;
                        }
                        if (uEntry.barycenter === undefined || vEntry.barycenter === undefined || uEntry.barycenter >= vEntry.barycenter) {
                            let sum = 0;
                            let weight = 0;
                            if (vEntry.weight) {
                                sum += vEntry.barycenter * vEntry.weight;
                                weight += vEntry.weight;
                            }
                            if (uEntry.weight) {
                                sum += uEntry.barycenter * uEntry.weight;
                                weight += uEntry.weight;
                            }
                            vEntry.vs = uEntry.vs.concat(vEntry.vs);
                            vEntry.barycenter = sum / weight;
                            vEntry.weight = weight;
                            vEntry.i = Math.min(uEntry.i, vEntry.i);
                            uEntry.merged = true;
                        }
                    };
                }
                function handleOut(vEntry) {
                    return function(wEntry) {
                        wEntry.in.push(vEntry);
                        if (--wEntry.indegree === 0) {
                            sourceSet.push(wEntry);
                        }
                    };
                }
                while (sourceSet.length) {
                    const entry = sourceSet.pop();
                    results.push(entry);
                    entry.in.reverse().forEach(handleIn(entry));
                    entry.out.forEach(handleOut(entry));
                }
                return results.filter((entry) => !entry.merged).map((entry) => {
                    const value = {
                        vs: entry.vs,
                        i: entry.i
                    };
                    if (entry.barycenter !== undefined) {
                        value.barycenter = entry.barycenter;
                    }
                    if (entry.weight !== undefined) {
                        value.weight = entry.weight;
                    }
                    return value;
                });
            };
            const barycenter = (g, movable) => {
                return (movable || []).map((v) => {
                    const inV = g.node(v).in;
                    if (!inV.length) {
                        return { v };
                    }
                    const result = inV.reduce((acc, e) => {
                        const edge = e.label;
                        const nodeU = e.vNode.label;
                        return {
                            sum: acc.sum + (edge.weight * nodeU.order),
                            weight: acc.weight + edge.weight
                        };
                    }, { sum: 0, weight: 0 });
                    return {
                        v,
                        barycenter: result.sum / result.weight,
                        weight: result.weight
                    };
                });
            };
            const sort = (entries, biasRight) => {
                const consumeUnsortable = (vs, unsortable, index) => {
                    let last = null;
                    while (unsortable.length && (last = unsortable[unsortable.length - 1]).i <= index) {
                        unsortable.pop();
                        vs.push(last.vs);
                        index++;
                    }
                    return index;
                };
                const compareWithBias = (bias) => {
                    return function(entryV, entryW) {
                        if (entryV.barycenter < entryW.barycenter) {
                            return -1;
                        } else if (entryV.barycenter > entryW.barycenter) {
                            return 1;
                        }
                        return bias ? entryW.i - entryV.i : entryV.i - entryW.i;
                    };
                };
                // partition
                const parts = { lhs: [], rhs: [] };
                for (const value of entries) {
                    if ('barycenter' in value) {
                        parts.lhs.push(value);
                    } else {
                        parts.rhs.push(value);
                    }
                }
                const sortable = parts.lhs;
                const unsortable = parts.rhs.sort((a, b) => -a.i + b.i);
                const vs = [];
                let sum = 0;
                let weight = 0;
                let vsIndex = 0;
                sortable.sort(compareWithBias(Boolean(biasRight)));
                vsIndex = consumeUnsortable(vs, unsortable, vsIndex);
                for (const entry of sortable) {
                    vsIndex += entry.vs.length;
                    vs.push(entry.vs);
                    sum += entry.barycenter * entry.weight;
                    weight += entry.weight;
                    vsIndex = consumeUnsortable(vs, unsortable, vsIndex);
                }
                const result = { vs: flat(vs) };
                if (weight) {
                    result.barycenter = sum / weight;
                    result.weight = weight;
                }
                return result;
            };
            const node = g.node(v);
            const bl = node && node.label ? node.label.borderLeft : undefined;
            const br = node && node.label ? node.label.borderRight : undefined;
            const subgraphs = {};
            const movable = bl ? g.children(v).filter((w) => w !== bl && w !== br) : g.children(v);
            const barycenters = barycenter(g, movable);
            for (const entry of barycenters) {
                if (g.children(entry.v).length) {
                    const result = sortSubgraph(g, entry.v, cg, biasRight);
                    subgraphs[entry.v] = result;
                    if ('barycenter' in result) {
                        if (entry.barycenter === undefined) {
                            entry.barycenter = result.barycenter;
                            entry.weight = result.weight;
                        } else {
                            entry.barycenter = (entry.barycenter * entry.weight + result.barycenter * result.weight) / (entry.weight + result.weight);
                            entry.weight += result.weight;
                        }
                    }
                }
            }
            const entries = resolveConflicts(barycenters, cg);
            // expand subgraphs
            for (const entry of entries) {
                entry.vs = flat(entry.vs.map((v) => subgraphs[v] ? subgraphs[v].vs : v));
            }
            const result = sort(entries, biasRight);
            if (bl) {
                result.vs = flat([bl, result.vs, br]);
                if (g.predecessors(bl).length) {
                    const blPred = g.node(g.predecessors(bl)[0]).label;
                    const brPred = g.node(g.predecessors(br)[0]).label;
                    if (!('barycenter' in result)) {
                        result.barycenter = 0;
                        result.weight = 0;
                    }
                    result.barycenter = (result.barycenter * result.weight + blPred.order + brPred.order) / (result.weight + 2);
                    result.weight += 2;
                }
            }
            return result;
        };
        const sweepLayerGraphs = (layerGraphs, biasRight) => {
            const cg = new dagre.Graph(true, false);
            for (const lg of layerGraphs) {
                const root = lg.root;
                const sorted = sortSubgraph(lg, root, cg, biasRight);
                const vs = sorted.vs;
                const length = vs.length;
                for (let i = 0; i < length; i++) {
                    lg.node(vs[i]).label.order = i;
                }
                // add subgraph constraints
                const prev = {};
                let rootPrev = '';
                let exit = false;
                for (const v of vs) {
                    let child = lg.parent(v);
                    let prevChild = null;
                    while (child) {
                        const parent = lg.parent(child);
                        if (parent) {
                            prevChild = prev[parent];
                            prev[parent] = child;
                        } else {
                            prevChild = rootPrev;
                            rootPrev = child;
                        }
                        if (prevChild && prevChild !== child) {
                            cg.setEdge(prevChild, child, null);
                            exit = true;
                            break;
                        }
                        child = parent;
                    }
                    if (exit) {
                        break;
                    }
                }
            }
        };
        // A function that takes a layering (an array of layers, each with an array of
        // ordererd nodes) and a graph and returns a weighted crossing count.
        //
        // Pre-conditions:
        //    1. Input graph must be simple (not a multigraph), directed, and include
        //       only simple edges.
        //    2. Edges in the input graph must have assigned weights.
        //
        // Post-conditions:
        //    1. The graph and layering matrix are left unchanged.
        //
        // This algorithm is derived from Barth, et al., 'Bilayer Cross Counting.'
        const crossCount = (g, layering) => {
            let count = 0;
            for (let i = 1; i < layering.length; i++) {
                const northLayer = layering[i - 1];
                const southLayer = layering[i];
                // Sort all of the edges between the north and south layers by their position in the north layer and then the south.
                // Map these edges to the position of their head in the south layer.
                const southPos = {};
                for (let i = 0; i < southLayer.length; i++) {
                    southPos[southLayer[i]] = i;
                }
                const southEntries = [];
                for (const v of northLayer) {
                    const entries = [];
                    for (const e of g.node(v).out) {
                        entries.push({
                            pos: southPos[e.w],
                            weight: e.label.weight
                        });
                    }
                    entries.sort((a, b) => a.pos - b.pos);
                    for (const entry of entries) {
                        southEntries.push(entry);
                    }
                }
                // Build the accumulator tree
                let firstIndex = 1;
                while (firstIndex < southLayer.length) {
                    firstIndex <<= 1;
                }
                const treeSize = 2 * firstIndex - 1;
                firstIndex -= 1;
                const tree = Array.from(new Array(treeSize), () => 0);
                // Calculate the weighted crossings
                for (const entry of southEntries) {
                    let index = entry.pos + firstIndex;
                    tree[index] += entry.weight;
                    let weightSum = 0;
                    while (index > 0) {
                        if (index % 2) {
                            weightSum += tree[index + 1];
                        }
                        index = (index - 1) >> 1;
                        tree[index] += entry.weight;
                    }
                    count += entry.weight * weightSum;
                }
            }
            return count;
        };
        // Assigns an initial order value for each node by performing a DFS search
        // starting from nodes in the first rank. Nodes are assigned an order in their
        // rank as they are first visited.
        //
        // This approach comes from Gansner, et al., 'A Technique for Drawing Directed
        // Graphs.'
        //
        // Returns a layering matrix with an array per layer and each layer sorted by
        // the order of its nodes.
        const initOrder = (g) => {
            const visited = new Set();
            const nodes = Array.from(g.nodes.values()).filter((node) => g.children(node.v).length === 0);
            let maxRank = -1;
            for (const node of nodes) {
                const rank = node.label.rank;
                if (maxRank === -1 || (rank !== undefined && rank > maxRank)) {
                    maxRank = rank;
                }
            }
            if (maxRank !== -1) {
                const layers = Array.from(new Array(maxRank + 1), () => []);
                const queue = nodes.sort((a, b) => a.label.rank - b.label.rank).map((node) => node.v).reverse();
                for (let i = 0; i < queue.length; i++) {
                    const v = queue[i];
                    if (!visited.has(v)) {
                        visited.add(v);
                        const rank = g.node(v).label.rank;
                        layers[rank].push(v);
                        for (const w of g.successors(v)) {
                            queue.push(w);
                        }
                    }
                }
                return layers;
            }
            return [];
        };
        // Constructs a graph that can be used to sort a layer of nodes.
        // The graph will contain all base and subgraph nodes from the request layer in their original
        // hierarchy and any edges that are incident on these nodes and are of the type requested by the 'relationship' parameter.
        //
        // Nodes from the requested rank that do not have parents are assigned a root node in the output graph,
        // which is set in the root graph attribute.
        // This makes it easy to walk the hierarchy of movable nodes during ordering.
        //
        // Pre-conditions:
        //    1. Input graph is a DAG
        //    2. Base nodes in the input graph have a rank attribute
        //    3. Subgraph nodes in the input graph has minRank and maxRank attributes
        //    4. Edges have an assigned weight
        //
        // Post-conditions:
        //    1. Output graph has all nodes in the movable rank with preserved hierarchy.
        //    2. Root nodes in the movable layer are made children of the node
        //       indicated by the root attribute of the graph.
        //    3. Non-movable nodes incident on movable nodes, selected by the
        //       relationship parameter, are included in the graph (without hierarchy).
        //    4. Edges incident on movable nodes, selected by the relationship parameter, are added to the output graph.
        //    5. The weights for copied edges are aggregated as need, since the output graph is not a multi-graph.
        const buildLayerGraph = (g, nodes, rank, relationship) => {
            let root = '';
            while (g.hasNode((root = uniqueId('_root')))) {
                // continue
            }
            const graph = new dagre.Graph(true, true);
            graph.root = root;
            graph.setDefaultNodeLabel((v) => {
                const node = g.node(v);
                return node ? node.label : undefined;
            });
            const length = nodes.length;
            let i = 0;
            while (i < length) {
                const node = nodes[i++];
                const label = node.label;
                if (label.rank === rank || 'minRank' in label && 'maxRank' in label && label.minRank <= rank && rank <= label.maxRank) {
                    const v = node.v;
                    graph.setNode(v);
                    const parent = g.parent(v);
                    graph.setParent(v, parent || root);
                    // This assumes we have only short edges!
                    if (relationship) {
                        for (const e of node.in) {
                            graph.setEdge(e.v, v, { weight: e.label.weight });
                        }
                    } else {
                        for (const e of node.out) {
                            graph.setEdge(e.w, v, { weight: e.label.weight });
                        }
                    }
                    if ('minRank' in label) {
                        graph.setNode(v, {
                            borderLeft: label.borderLeft[rank],
                            borderRight: label.borderRight[rank]
                        });
                    }
                }
            }
            return graph;
        };
        let layering = initOrder(g);
        const assignOrder = (g, layering) => {
            for (const layer of layering) {
                for (let i = 0; i < layer.length; i++) {
                    g.node(layer[i]).label.order = i;
                }
            }
        };
        assignOrder(g, layering);
        const rank = maxRank(g) || 0;
        const downLayerGraphs = new Array(rank);
        const upLayerGraphs = new Array(rank);
        const nodes = Array.from(g.nodes.values());
        for (let i = 0; i < rank; i++) {
            downLayerGraphs[i] = buildLayerGraph(g, nodes, i + 1, true);
            upLayerGraphs[i] = buildLayerGraph(g, nodes, rank - i - 1, false);
        }
        let bestCC = Number.POSITIVE_INFINITY;
        let best = [];
        for (let i = 0, lastBest = 0; lastBest < 4; ++i, ++lastBest) {
            sweepLayerGraphs(i % 2 ? downLayerGraphs : upLayerGraphs, i % 4 >= 2);
            layering = buildLayerMatrix(g);
            const cc = crossCount(g, layering);
            if (cc < bestCC) {
                lastBest = 0;
                const length = layering.length;
                best = new Array(length);
                for (let j = 0; j < length; j++) {
                    best[j] = layering[j].slice();
                }
                bestCC = cc;
            }
        }
        // Reduce crossings
        const exchange = (layer, node0, node1) => {
            const index0 = layer.indexOf(node0.v);
            const index1 = layer.indexOf(node1.v);
            layer[index1] = node0.v;
            layer[index0] = node1.v;
        };
        for (let i = 0; i < best.length - 2; i += 2) {
            const layer0 = best[i];
            const layer1 = best[i + 1];
            const layer2 = best[i + 2];
            for (let j = 0; j < layer2.length; ++j) {
                const node0 = g.nodes.get(layer2[j]);
                if (node0.in && node0.in.length >= 2) {
                    for (let k = 0; k < node0.in.length - 1; ++k) {
                        const node1d = node0.in[k].vNode;
                        const node2d = node0.in[k + 1].vNode;
                        const node1 = node1d.in[0].vNode;
                        const node2 = node2d.in[0].vNode;
                        if ((layer1.indexOf(node1d.v) < layer1.indexOf(node2d.v)) ^ (layer0.indexOf(node1.v) < layer0.indexOf(node2.v))) {
                            exchange(layer1, node1d, node2d);
                        }
                    }
                }
            }
        }
        for (let i = 0; i < best.length - 4; i += 2) {
            const layer0 = best[i];
            const layer2 = best[i + 2];
            const layer4 = best[i + 4];
            if (layer2.length >= 2 && layer4.length >= 2) {
                const layer1 = best[i + 1];
                const layer3 = best[i + 3];
                for (let j = 0; j < layer0.length; ++j) {
                    const node0 = g.nodes.get(layer0[j]);
                    if (node0.in && node0.out && node0.out.length >= 2) {
                        for (let k = 0; k < node0.out.length - 1; ++k) {
                            const node1u = node0.out[k].wNode;
                            const node2u = node0.out[k + 1].wNode;
                            const node1 = node1u.out[0].wNode;
                            const node2 = node2u.out[0].wNode;
                            if (node1.out.length === 1 && node2.out.length === 1) {
                                const index1 = layer2.indexOf(node1.v);
                                const index2 = layer2.indexOf(node2.v);
                                if (index1 + 1 === index2) {
                                    const node1d = node1.out[0].wNode;
                                    const node2d = node2.out[0].wNode;
                                    if (node1d.out.length === 1 && node2d.out.length === 1) {
                                        const node3 = node1d.out[0].wNode;
                                        const node4 = node2d.out[0].wNode;
                                        const index3 = layer4.indexOf(node3.v);
                                        const index4 = layer4.indexOf(node4.v);
                                        if (index3 > index4) {
                                            exchange(layer1, node1u, node2u);
                                            exchange(layer2, node1, node2);
                                            exchange(layer3, node1d, node2d);
                                            ++k;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                for (let j = 0; j < layer2.length - 1; ++j) {
                    const node0 = g.nodes.get(layer2[j]);
                    if (node0.in && node0.out && node0.in.length === 1 && node0.out.length === 1) {
                        const node1 = g.nodes.get(layer2[j + 1]);
                        if (node1.in && node1.out && node1.in.length === 1 && node1.out.length === 1) {
                            const node0u = node0.in[0].vNode;
                            const node1u = node1.in[0].vNode;
                            if (node0u.in.length === 1 && node1u.in.length === 1) {
                                const node2 = node0u.in[0].vNode;
                                const node3 = node1u.in[0].vNode;
                                let index0 = layer0.indexOf(node2.v);
                                let index1 = layer0.indexOf(node3.v);
                                if (index1 + 1 === index0) {
                                    const node0d = node0.out[0].wNode;
                                    const node1d = node1.out[0].wNode;
                                    index0 = layer3.indexOf(node0d.v);
                                    index1 = layer3.indexOf(node1d.v);
                                    if (index0 + 1 === index1 && node0d.out[0].wNode === node1d.out[0].wNode) {
                                        exchange(layer1, node0u, node1u);
                                        exchange(layer2, node0, node1);
                                        exchange(layer3, node0d, node1d);
                                        j += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assignOrder(g, best);
    };

    const insertSelfEdges = (g) => {
        const layers = buildLayerMatrix(g);
        for (const layer of layers) {
            let orderShift = 0;
            layer.forEach((v, i) => {
                const label = g.node(v).label;
                label.order = i + orderShift;
                if (label.selfEdges) {
                    for (const selfEdge of label.selfEdges) {
                        addDummyNode(g, 'selfedge', {
                            width: selfEdge.label.width,
                            height: selfEdge.label.height,
                            rank: label.rank,
                            order: i + (++orderShift),
                            e: selfEdge.e,
                            label: selfEdge.label
                        }, '_se');
                    }
                    delete label.selfEdges;
                }
            });
        }
    };

    const coordinateSystem_swapWidthHeight = (g) => {
        for (const node of g.nodes.values()) {
            const label = node.label;
            const w = label.width;
            label.width = label.height;
            label.height = w;
        }
        for (const e of g.edges.values()) {
            const label = e.label;
            const w = label.width;
            label.width = label.height;
            label.height = w;
        }
    };

    const coordinateSystem_adjust = (g, state, layout) => {
        const rankDir = layout.rankdir.toLowerCase();
        if (rankDir === 'lr' || rankDir === 'rl') {
            coordinateSystem_swapWidthHeight(g);
        }
    };

    const coordinateSystem_undo = (g, state, layout) => {
        const rankDir = layout.rankdir.toLowerCase();
        if (rankDir === 'bt' || rankDir === 'rl') {
            for (const node of g.nodes.values()) {
                node.label.y = -node.label.y;
            }
            for (const e of g.edges.values()) {
                const edge = e.label;
                for (const attr of edge.points) {
                    attr.y = -attr.y;
                }
                if ('y' in edge) {
                    edge.y = -edge.y;
                }
            }
        }
        if (rankDir === 'lr' || rankDir === 'rl') {
            const swapXYOne = (attrs) => {
                const x = attrs.x;
                attrs.x = attrs.y;
                attrs.y = x;
            };
            for (const node of g.nodes.values()) {
                swapXYOne(node.label);
            }
            for (const e of g.edges.values()) {
                const edge = e.label;
                for (const e of edge.points) {
                    swapXYOne(e);
                }
                if (edge.x !== undefined) {
                    swapXYOne(edge);
                }
            }
            coordinateSystem_swapWidthHeight(g);
        }
    };

    const position = (g, state, layout) => {
        const addConflict = (conflicts, v, w) => {
            if (v > w) {
                const tmp = v;
                v = w;
                w = tmp;
            }
            let conflictsV = conflicts[v];
            if (!conflictsV) {
                conflictsV = {};
                conflicts[v] = conflictsV;
            }
            conflictsV[w] = true;
        };
        const hasConflict = (conflicts, v, w) => {
            if (v > w) {
                const tmp = v;
                v = w;
                w = tmp;
            }
            return conflicts[v] && w in conflicts[v];
        };
        const buildBlockGraph = (g, layout, layering, root, reverseSep) => {
            const nodeSep = layout.nodesep;
            const edgeSep = layout.edgesep;
            const blockGraph = new dagre.Graph(true, false);
            for (const layer of layering) {
                let u = null;
                for (const v of layer) {
                    const vRoot = root[v];
                    blockGraph.setNode(vRoot, {});
                    if (u) {
                        const uRoot = root[u];
                        const vLabel = g.node(v).label;
                        const wLabel = g.node(u).label;
                        let sum = 0;
                        let delta = 0;
                        sum += vLabel.width / 2;
                        if ('labelpos' in vLabel) {
                            switch (vLabel.labelpos) {
                                case 'l': delta = -vLabel.width / 2; break;
                                case 'r': delta = vLabel.width / 2; break;
                                default: throw new dagre.Error(`Unsupported label position '${vLabel.labelpos}'.`);
                            }
                        }
                        if (delta) {
                            sum += reverseSep ? delta : -delta;
                        }
                        delta = 0;
                        sum += (vLabel.dummy ? edgeSep : nodeSep) / 2;
                        sum += (wLabel.dummy ? edgeSep : nodeSep) / 2;
                        sum += wLabel.width / 2;
                        if ('labelpos' in wLabel) {
                            switch (wLabel.labelpos) {
                                case 'l': delta = wLabel.width / 2; break;
                                case 'r': delta = -wLabel.width / 2; break;
                                default: throw new dagre.Error(`Unsupported label position '${wLabel.labelpos}'.`);
                            }
                        }
                        if (delta) {
                            sum += reverseSep ? delta : -delta;
                        }
                        const edge = blockGraph.edge(uRoot, vRoot);
                        const max = Math.max(sum, edge ? edge.label : 0);
                        if (edge) {
                            edge.label = max;
                        } else {
                            blockGraph.setEdge(uRoot, vRoot, max);
                        }
                    }
                    u = v;
                }
            }
            return blockGraph;
        };
        // Try to align nodes into vertical 'blocks' where possible.
        // This algorithm attempts to align a node with one of its median neighbors.
        // If the edge connecting a neighbor is a type-1 conflict then we ignore that possibility.
        // If a previous node has already formed a block with a node after the node we're trying to form a block with,
        // we also ignore that possibility - our blocks would be split in that scenario.
        const verticalAlignment = (layering, conflicts, neighborFn) => {
            const root = {};
            const align = {};
            const pos = {};
            // We cache the position here based on the layering because the graph and layering may be out of sync.
            // The layering matrix is manipulated to generate different extreme alignments.
            for (const layer of layering) {
                let order = 0;
                for (const v of layer) {
                    root[v] = v;
                    align[v] = v;
                    pos[v] = order;
                    order++;
                }
            }
            for (const layer of layering) {
                let prevIdx = -1;
                for (const v of layer) {
                    let ws = neighborFn(v);
                    if (ws.length > 0) {
                        ws = ws.sort((a, b) => pos[a] - pos[b]);
                        const mp = (ws.length - 1) / 2.0;
                        const il = Math.ceil(mp);
                        for (let i = Math.floor(mp); i <= il; i++) {
                            const w = ws[i];
                            if (align[v] === v && prevIdx < pos[w] && !hasConflict(conflicts, v, w)) {
                                const x = root[w];
                                align[w] = v;
                                align[v] = x;
                                root[v] = x;
                                prevIdx = pos[w];
                            }
                        }
                    }
                }
            }
            return { root, align };
        };
        const horizontalCompaction = (g, layout, layering, root, align, reverseSep) => {
            // This portion of the algorithm differs from BK due to a number of problems.
            // Instead of their algorithm we construct a new block graph and do two sweeps.
            const blockG = buildBlockGraph(g, layout, layering, root, reverseSep);
            const borderType = reverseSep ? 'borderLeft' : 'borderRight';
            const xs = {};
            // First pass, places blocks with the smallest possible coordinates.
            if (blockG.nodes.size > 0) {
                const stack = Array.from(blockG.nodes.keys());
                const visited = new Set();
                while (stack.length > 0) {
                    const v = stack.pop();
                    if (visited.has(v)) {
                        let max = 0;
                        for (const e of blockG.node(v).in) {
                            max = Math.max(max, xs[e.v] + e.label);
                        }
                        xs[v] = max;
                    } else {
                        visited.add(v);
                        stack.push(v);
                        for (const w of blockG.predecessors(v)) {
                            stack.push(w);
                        }
                    }
                }
            }
            // Second pass, removes unused space by moving blocks to the greatest coordinates without violating separation.
            if (blockG.nodes.size > 0) {
                const stack = Array.from(blockG.nodes.keys());
                const visited = new Set();
                while (stack.length > 0) {
                    const v = stack.pop();
                    if (visited.has(v)) {
                        let min = Number.POSITIVE_INFINITY;
                        for (const e of blockG.node(v).out) {
                            min = Math.min(min, xs[e.w] - e.label);
                        }
                        const label = g.node(v).label;
                        if (min !== Number.POSITIVE_INFINITY && label.borderType !== borderType) {
                            xs[v] = Math.max(xs[v], min);
                        }
                    } else {
                        visited.add(v);
                        stack.push(v);
                        for (const w of blockG.successors(v)) {
                            stack.push(w);
                        }
                    }
                }
            }
            // Assign x coordinates to all nodes
            for (const v of Object.values(align)) {
                xs[v] = xs[root[v]];
            }
            return xs;
        };
        // Marks all edges in the graph with a type-1 conflict with the 'type1Conflict' property.
        // A type-1 conflict is one where a non-inner segment crosses an inner segment.
        // An inner segment is an edge with both incident nodes marked with the 'dummy' property.
        //
        // This algorithm scans layer by layer, starting with the second, for type-1
        // conflicts between the current layer and the previous layer. For each layer
        // it scans the nodes from left to right until it reaches one that is incident
        // on an inner segment. It then scans predecessors to determine if they have
        // edges that cross that inner segment. At the end a final scan is done for all
        // nodes on the current rank to see if they cross the last visited inner segment.
        //
        // This algorithm (safely) assumes that a dummy node will only be incident on a
        // single node in the layers being scanned.
        const findType1Conflicts = (g, layering) => {
            const conflicts = {};
            if (layering.length > 0) {
                let [prev] = layering;
                for (let k = 1; k < layering.length; k++) {
                    const layer = layering[k];
                    // last visited node in the previous layer that is incident on an inner segment.
                    let k0 = 0;
                    // Tracks the last node in this layer scanned for crossings with a type-1 segment.
                    let scanPos = 0;
                    const prevLayerLength = prev.length;
                    const lastNode = layer[layer.length - 1];
                    for (let i = 0; i < layer.length; i++) {
                        const v = layer[i];
                        const w = g.node(v).label.dummy ? g.predecessors(v).find((u) => g.node(u).label.dummy) : null;
                        if (w || v === lastNode) {
                            const k1 = w ? g.node(w).label.order : prevLayerLength;
                            for (const scanNode of layer.slice(scanPos, i + 1)) {
                            // for (const scanNode of layer.slice(scanPos, scanPos + 1)) {
                                for (const u of g.predecessors(scanNode)) {
                                    const uLabel = g.node(u).label;
                                    const uPos = uLabel.order;
                                    if ((uPos < k0 || k1 < uPos) && !(uLabel.dummy && g.node(scanNode).label.dummy)) {
                                        addConflict(conflicts, u, scanNode);
                                    }
                                }
                            }
                            // scanPos += 1;
                            scanPos = i + 1;
                            k0 = k1;
                        }
                    }
                    prev = layer;
                }
            }
            return conflicts;
        };
        const findType2Conflicts = (g, layering) => {
            const conflicts = {};
            const scan = (south, southPos, southEnd, prevNorthBorder, nextNorthBorder) => {
                for (let i = southPos; i < southEnd; i++) {
                    const v = south[i];
                    if (g.node(v).labeldummy) {
                        for (const u of g.predecessors(v)) {
                            const uNode = g.node(u).label;
                            if (uNode.dummy && (uNode.order < prevNorthBorder || uNode.order > nextNorthBorder)) {
                                addConflict(conflicts, u, v);
                            }
                        }
                    }
                }
            };
            if (layering.length > 0) {
                let [north] = layering;
                for (let i = 1; i < layering.length; i++) {
                    const south = layering[i];
                    let prevNorthPos = -1;
                    let nextNorthPos = 0;
                    let southPos = 0;
                    south.forEach((v, southLookahead) => {
                        if (g.node(v).label.dummy === 'border') {
                            const predecessors = g.predecessors(v);
                            if (predecessors.length) {
                                nextNorthPos = g.node(predecessors[0]).label.order;
                                scan(south, southPos, southLookahead, prevNorthPos, nextNorthPos);
                                southPos = southLookahead;
                                prevNorthPos = nextNorthPos;
                            }
                        }
                        scan(south, southPos, south.length, nextNorthPos, north.length);
                    });
                    north = south;
                }
            }
            return conflicts;
        };

        g = asNonCompoundGraph(g);
        const layering = buildLayerMatrix(g);
        const ranksep = layout.ranksep;
        // Assign y-coordinate based on rank
        let y = 0;
        for (const layer of layering) {
            const maxHeight = layer.reduce((a, v) => Math.max(a, g.node(v).label.height), 0);
            for (const v of layer) {
                g.node(v).label.y = y + maxHeight / 2;
            }
            y += maxHeight + ranksep;
        }
        // Coordinate assignment based on Brandes and Köpf, 'Fast and Simple Horizontal Coordinate Assignment.'
        const conflicts = Object.assign(findType1Conflicts(g, layering), findType2Conflicts(g, layering));
        const xss = {};
        for (const vertical of ['u', 'd']) {
            let adjustedLayering = vertical === 'u' ? layering : Object.values(layering).reverse();
            for (const horizontal of ['l', 'r']) {
                if (horizontal === 'r') {
                    adjustedLayering = adjustedLayering.map((layer) => Object.values(layer).reverse());
                }
                const neighborFn = (vertical === 'u' ? g.predecessors : g.successors).bind(g);
                const align = verticalAlignment(adjustedLayering, conflicts, neighborFn);
                const xs = horizontalCompaction(g, layout, adjustedLayering, align.root, align.align, horizontal === 'r');
                if (horizontal === 'r') {
                    for (const entry of Object.entries(xs)) {
                        xs[entry[0]] = -entry[1];
                    }
                }
                xss[vertical + horizontal] = xs;
            }
        }
        // Find smallest width alignment: Returns the alignment that has the smallest width of the given alignments.
        let minWidth = Number.POSITIVE_INFINITY;
        let minValue = null;
        for (const xs of Object.values(xss)) {
            let max = Number.NEGATIVE_INFINITY;
            let min = Number.POSITIVE_INFINITY;
            for (const [v, x] of Object.entries(xs)) {
                const halfWidth = g.node(v).label.width / 2;
                max = Math.max(x + halfWidth, max);
                min = Math.min(x - halfWidth, min);
            }
            const width = max - min;
            if (width < minWidth) {
                minWidth = width;
                minValue = xs;
            }
        }
        // Align the coordinates of each of the layout alignments such that
        // left-biased alignments have their minimum coordinate at the same point as
        // the minimum coordinate of the smallest width alignment and right-biased
        // alignments have their maximum coordinate at the same point as the maximum
        // coordinate of the smallest width alignment.
        const alignTo = minValue;
        const range = (values) => {
            let min = Number.POSITIVE_INFINITY;
            let max = Number.NEGATIVE_INFINITY;
            for (const value of values) {
                if (value < min) {
                    min = value;
                }
                if (value > max) {
                    max = value;
                }
            }
            return [min, max];
        };
        const alignToRange = range(Object.values(alignTo));
        for (const vertical of ['u', 'd']) {
            for (const horizontal of ['l', 'r']) {
                const alignment = vertical + horizontal;
                const xs = xss[alignment];
                if (xs !== alignTo) {
                    const vsValsRange = range(Object.values(xs));
                    const delta = horizontal === 'l' ? alignToRange[0] - vsValsRange[0] : alignToRange[1] - vsValsRange[1];
                    if (delta) {
                        const list = {};
                        for (const key of Object.keys(xs)) {
                            list[key] = xs[key] + delta;
                        }
                        xss[alignment] = list;
                    }
                }
            }
        }
        // balance
        const align = layout.align;
        if (align) {
            const xs = xss[align.toLowerCase()];
            for (const v of Object.keys(xss.ul)) {
                g.node(v).label.x = xs[v];
            }
        } else {
            for (const v of Object.keys(xss.ul)) {
                const xs = [xss.ul[v], xss.ur[v], xss.dl[v], xss.dr[v]].sort((a, b) => a - b);
                g.node(v).label.x = (xs[1] + xs[2]) / 2;
            }
        }
    };

    const positionSelfEdges = (g) => {
        for (const node of g.nodes.values()) {
            const label = node.label;
            if (label.dummy === 'selfedge') {
                const v = node.v;
                const selfNode = g.node(label.e.v).label;
                const x = selfNode.x + selfNode.width / 2;
                const y = selfNode.y;
                const dx = label.x - x;
                const dy = selfNode.height / 2;
                g.setEdge(label.e.v, label.e.w, label.label);
                g.removeNode(v);
                label.label.points = [
                    { x: x + 2 * dx / 3, y: y - dy },
                    { x: x + 5 * dx / 6, y: y - dy },
                    { x: x +     dx    , y },
                    { x: x + 5 * dx / 6, y: y + dy },
                    { x: x + 2 * dx / 3, y: y + dy }
                ];
                label.label.x = label.x;
                label.label.y = label.y;
            }
        }
    };

    const removeBorderNodes = (g) => {
        for (const node of g.nodes.values()) {
            const v = node.v;
            if (g.children(v).length) {
                const label = node.label;
                const t = g.node(label.borderTop).label;
                const b = g.node(label.borderBottom).label;
                const l = g.node(label.borderLeft[label.borderLeft.length - 1]).label;
                const r = g.node(label.borderRight[label.borderRight.length - 1]).label;
                label.width = Math.abs(r.x - l.x);
                label.height = Math.abs(b.y - t.y);
                label.x = l.x + label.width / 2;
                label.y = t.y + label.height / 2;
            }
        }
        for (const node of g.nodes.values()) {
            if (node.label.dummy === 'border') {
                g.removeNode(node.v);
            }
        }
    };

    const fixupEdgeLabelCoords = (g) => {
        for (const e of g.edges.values()) {
            const edge = e.label;
            if ('x' in edge) {
                if (edge.labelpos === 'l' || edge.labelpos === 'r') {
                    edge.width -= edge.labeloffset;
                }
                switch (edge.labelpos) {
                    case 'l': edge.x -= edge.width / 2 + edge.labeloffset; break;
                    case 'r': edge.x += edge.width / 2 + edge.labeloffset; break;
                    default: throw new dagre.Error(`Unsupported label position '${edge.labelpos}'.`);
                }
            }
        }
    };

    const translateGraph = (g, state) => {
        let minX = Number.POSITIVE_INFINITY;
        let maxX = 0;
        let minY = Number.POSITIVE_INFINITY;
        let maxY = 0;
        const getExtremes = (attrs) => {
            const x = attrs.x;
            const y = attrs.y;
            const w = attrs.width;
            const h = attrs.height;
            minX = Math.min(minX, x - w / 2);
            maxX = Math.max(maxX, x + w / 2);
            minY = Math.min(minY, y - h / 2);
            maxY = Math.max(maxY, y + h / 2);
        };
        for (const node of g.nodes.values()) {
            getExtremes(node.label);
        }
        for (const e of g.edges.values()) {
            const edge = e.label;
            if ('x' in edge) {
                getExtremes(edge);
            }
        }
        for (const node of g.nodes.values()) {
            node.label.x -= minX;
            node.label.y -= minY;
        }
        for (const e of g.edges.values()) {
            const edge = e.label;
            for (const p of edge.points) {
                p.x -= minX;
                p.y -= minY;
            }
            if ('x' in edge) {
                edge.x -= minX;
            }
            if ('y' in edge) {
                edge.y -= minY;
            }
        }
        state.width = maxX - minX;
        state.height = maxY - minY;
    };

    const assignNodeIntersects = (g) => {
        // Finds where a line starting at point ({x, y}) would intersect a rectangle
        // ({x, y, width, height}) if it were pointing at the rectangle's center.
        const intersectRect = (rect, point) => {
            const x = rect.x;
            const y = rect.y;
            // Rectangle intersection algorithm from: http://math.stackexchange.com/questions/108113/find-edge-between-two-boxes
            const dx = point.x - x;
            const dy = point.y - y;
            if (dx === 0 && dy === 0) {
                throw new Error('Not possible to find intersection inside of the rectangle');
            }
            let w = rect.width / 2;
            let h = rect.height / 2;
            if (Math.abs(dy) * w > Math.abs(dx) * h) {
                // Intersection is top or bottom of rect.
                h = dy < 0 ? -h : h;
                return { x: x + (h * dx / dy), y: y + h };
            }
            // Intersection is left or right of rect.
            w = dx < 0 ? -w : w;
            return { x: x + w, y: y + (w * dy / dx) };
        };
        for (const e of g.edges.values()) {
            const edge = e.label;
            const vNode = e.vNode.label;
            const wNode = e.wNode.label;
            let p1 = null;
            let p2 = null;
            if (edge.points) {
                [p1] = edge.points;
                p2 = edge.points[edge.points.length - 1];
            } else {
                edge.points = [];
                p1 = wNode;
                p2 = vNode;
            }
            edge.points.unshift(intersectRect(vNode, p1));
            edge.points.push(intersectRect(wNode, p2));
        }
    };

    // Build layout graph
    const g = new dagre.Graph(true, true);
    for (const node of nodes) {
        g.setNode(node.v, {
            width: node.width,
            height: node.height
        });
        if (node.parent) {
            g.setParent(node.v, node.parent);
        }
    }
    for (const edge of edges) {
        g.setEdge(edge.v, edge.w, {
            minlen: edge.minlen || 1,
            weight: edge.weight || 1,
            width: edge.width || 0,
            height: edge.height || 0,
            labeloffset: edge.labeloffset || 10,
            labelpos: edge.labelpos || 'r'
        });
    }

    // Run layout
    layout = { ranksep: 50, edgesep: 20, nodesep: 50, rankdir: 'tb', ...layout };
    const tasks = [
        makeSpaceForEdgeLabels,
        removeSelfEdges,
        acyclic_run,
        nestingGraph_run,
        rank,
        injectEdgeLabelProxies,
        removeEmptyRanks,
        nestingGraph_cleanup,
        assignRankMinMax,
        removeEdgeLabelProxies,
        normalize,
        parentDummyChains,
        addBorderSegments,
        order,
        insertSelfEdges,
        coordinateSystem_adjust,
        position,
        positionSelfEdges,
        removeBorderNodes,
        denormalize,
        fixupEdgeLabelCoords,
        coordinateSystem_undo,
        translateGraph,
        assignNodeIntersects,
        acyclic_undo
    ];
    while (tasks.length > 0) {
        // const start = Date.now();
        const task = tasks.shift();
        task(g, state, layout);
        // const duration = Date.now() - start;
        // console.log(`${task.name}: ${duration}ms`);
    }

    // Update source graph
    for (const node of nodes) {
        const label = g.node(node.v).label;
        node.x = label.x;
        node.y = label.y;
        if (g.children(node.v).length) {
            node.width = label.width;
            node.height = label.height;
        }
    }
    for (const edge of edges) {
        const label = g.edge(edge.v, edge.w).label;
        edge.points = label.points;
        if ('x' in label) {
            edge.x = label.x;
            edge.y = label.y;
        }
    }
};

dagre.Graph = class {

    constructor(directed, compound) {
        this.directed = directed;
        this.compound = compound;
        this._defaultNodeLabelFn = () => {
            return undefined;
        };
        this.nodes = new Map();
        this.edges = new Map();
        if (this.compound) {
            this._parent = {};
            this._children = {};
            this._children['\x00'] = {};
        }
    }

    setDefaultNodeLabel(newDefault) {
        this._defaultNodeLabelFn = newDefault;
    }

    setNode(v, label) {
        const node = this.nodes.get(v);
        if (node) {
            if (label) {
                node.label = label;
            }
        } else {
            const node = { label: label ? label : this._defaultNodeLabelFn(v), in: [], out: [], predecessors: {}, successors: {}, v };
            this.nodes.set(v, node);
            if (this.compound) {
                this._parent[v] = '\x00';
                this._children[v] = {};
                this._children['\x00'][v] = true;
            }
        }
    }

    node(v) {
        return this.nodes.get(v);
    }

    hasNode(v) {
        return this.nodes.has(v);
    }

    removeNode(v) {
        const node = this.nodes.get(v);
        if (node) {
            if (this.compound) {
                delete this._children[this._parent[v]][v];
                delete this._parent[v];
                for (const child of this.children(v)) {
                    this.setParent(child);
                }
                delete this._children[v];
            }
            for (const edge of node.in) {
                this.removeEdge(edge);
            }
            for (const edge of node.out) {
                this.removeEdge(edge);
            }
            this.nodes.delete(v);
        }
    }

    setParent(v, parent) {
        if (!this.compound) {
            throw new Error('Cannot set parent in a non-compound graph');
        }
        if (parent) {
            for (let ancestor = parent; ancestor !== undefined; ancestor = this.parent(ancestor)) {
                if (ancestor === v) {
                    throw new Error(`Setting ${parent} as parent of ${v} would create a cycle.`);
                }
            }
            this.setNode(parent);
        } else {
            parent = '\x00';
        }
        delete this._children[this._parent[v]][v];
        this._parent[v] = parent;
        this._children[parent][v] = true;
    }

    parent(v) {
        if (this.compound) {
            const parent = this._parent[v];
            if (parent !== '\x00') {
                return parent;
            }
        }
        return null;
    }

    children(v) {
        if (this.compound) {
            return Object.keys(this._children[v === undefined ? '\x00' : v]);
        } else if (v === undefined) {
            return this.nodes.keys();
        } else if (this.hasNode(v)) {
            return [];
        }
        return null;
    }

    predecessors(v) {
        return Object.keys(this.nodes.get(v).predecessors);
    }

    successors(v) {
        return Object.keys(this.nodes.get(v).successors);
    }

    neighbors(v) {
        return Array.from(new Set(this.predecessors(v).concat(this.successors(v))));
    }

    edge(v, w) {
        return this.edges.get(this._edgeKey(this.directed, v, w));
    }

    setEdge(v, w, label, name) {
        const key = this._edgeKey(this.directed, v, w, name);
        const edge = this.edges.get(key);
        if (edge) {
            edge.label = label;
        } else {
            if (!this.directed && v > w) {
                const tmp = v;
                v = w;
                w = tmp;
            }
            const edge = { label, v, w, name, key, vNode: null, wNode: null };
            this.edges.set(key, edge);
            this.setNode(v);
            this.setNode(w);
            const wNode = this.nodes.get(w);
            const vNode = this.nodes.get(v);
            edge.wNode = wNode;
            edge.vNode = vNode;
            const incrementOrInitEntry = (map, k) => {
                if (map[k]) {
                    map[k]++;
                } else {
                    map[k] = 1;
                }
            };
            incrementOrInitEntry(wNode.predecessors, v);
            incrementOrInitEntry(vNode.successors, w);
            wNode.in.push(edge);
            vNode.out.push(edge);
        }
    }

    removeEdge(edge) {
        const key = edge.key;
        const v = edge.v;
        const w = edge.w;
        const wNode = edge.wNode;
        const vNode = edge.vNode;
        if (--wNode.predecessors[v] === 0) {
            delete wNode.predecessors[v];
        }
        if (--vNode.successors[w] === 0) {
            delete vNode.successors[w];
        }
        wNode.in = wNode.in.filter((edge) => edge.key !== key);
        vNode.out = vNode.out.filter((edge) => edge.key !== key);
        this.edges.delete(key);
    }

    _edgeKey(isDirected, v, w, name) {
        if (!isDirected && v > w) {
            return name ? `${w}:${v}:${name}` : `${w}:${v}:`;
        }
        return name ? `${v}:${w}:${name}` : `${v}:${w}:`;
    }

    toString() {
        return [
            '[nodes]', Array.from(this.nodes.values()).map((n) => JSON.stringify(n.label)).join('\n'),
            '[edges]', Array.from(this.edges.values()).map((e) => JSON.stringify(e.label)).join('\n'),
            '[parents]', JSON.stringify(this._parent, null, 2),
            '[children]', JSON.stringify(this._children, null, 2)
        ].join('\n');
    }
};

export const { layout, Graph } = dagre;
