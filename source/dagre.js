
var dagre = dagre || {};

// Dagre graph layout
// https://github.com/dagrejs/dagre
// https://github.com/dagrejs/graphlib

dagre.layout = (graph, options) => {
    options = options || {};
    // options.time = true;
    const time = (name, callback) => {
        const start = Date.now();
        const result = callback();
        const duration = Date.now() - start;
        if (options.time) {
            console.log(name + ': ' + duration + 'ms');
        }
        return result;
    };

    // Constructs a new graph from the input graph, which can be used for layout.
    // This process copies only whitelisted attributes from the input graph to the
    // layout graph. Thus this function serves as a good place to determine what
    // attributes can influence layout.
    const buildLayoutGraph = (graph) => {
        const g = new dagre.Graph({ multigraph: true, compound: true });
        g.setGraph(Object.assign({}, { ranksep: 50, edgesep: 20, nodesep: 50, rankdir: 'tb' }, graph.graph()));
        for (const v of graph.nodes()) {
            const node = graph.node(v);
            g.setNode(v, {
                width: node.width || 0,
                height: node.height || 0
            });
            g.setParent(v, graph.parent(v));
        }
        for (const e of graph.edges()) {
            const edge = graph.edge(e);
            g.setEdge(e, {
                minlen: edge.minlin || 1,
                weight: edge.weight || 1,
                width: edge.width || 0,
                height: edge.height || 0,
                labeloffset: edge.labeloffset || 10,
                labelpos: edge.labelpos || 'r'
            });
        }
        return g;
    };

    const runLayout = (g, time) => {

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
        const util_addDummyNode = (g, type, attrs, name) => {
            let v;
            do {
                v = uniqueId(name);
            } while (g.hasNode(v));

            attrs.dummy = type;
            g.setNode(v, attrs);
            return v;
        };

        const util_asNonCompoundGraph = (g) => {
            const graph = new dagre.Graph({ multigraph: g.isMultigraph() });
            graph.setGraph(g.graph());
            for (const v of g.nodes()) {
                if (g.children(v).length === 0) {
                    graph.setNode(v, g.node(v));
                }
            }
            for (const e of g.edges()) {
                graph.setEdge(e, g.edge(e));
            }
            return graph;
        };

        const maxRank = (g) => {
            let rank = Number.NEGATIVE_INFINITY;
            for (const v of g.nodes()) {
                const x = g.node(v).rank;
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
            for (const v of g.nodes()) {
                const node = g.node(v);
                const rank = node.rank;
                if (rank !== undefined) {
                    layering[rank][node.order] = v;
                }
            }
            return layering;
        };

        /*
        * This idea comes from the Gansner paper: to account for edge labels in our
        * layout we split each rank in half by doubling minlen and halving ranksep.
        * Then we can place labels at these mid-points between nodes.
        *
        * We also add some minimal padding to the width to push the label for the edge
        * away from the edge itself a bit.
        */
        const makeSpaceForEdgeLabels = (g) => {
            const graph = g.graph();
            graph.ranksep /= 2;
            for (const e of g.edges()) {
                const edge = g.edge(e);
                edge.minlen *= 2;
                if (edge.labelpos.toLowerCase() !== 'c') {
                    if (graph.rankdir === 'TB' || graph.rankdir === 'BT') {
                        edge.width += edge.labeloffset;
                    }
                    else {
                        edge.height += edge.labeloffset;
                    }
                }
            }
        };

        /*
        * A helper that preforms a pre- or post-order traversal on the input graph
        * and returns the nodes in the order they were visited. If the graph is
        * undirected then this algorithm will navigate using neighbors. If the graph
        * is directed then this algorithm will navigate using successors.
        *
        * Order must be one of 'pre' or 'post'.
        */
        const dfs = (g, vs, order) => {
            const doDfs = (g, v, postorder, visited, navigation, acc) => {
                if (!(v in visited)) {
                    visited[v] = true;
                    if (!postorder) {
                        acc.push(v);
                    }
                    for (const w of navigation(v)) {
                        doDfs(g, w, postorder, visited, navigation, acc);
                    }
                    if (postorder) {
                        acc.push(v);
                    }
                }
            };
            if (!Array.isArray(vs)) {
                vs = [ vs ];
            }
            const navigation = (g.isDirected() ? g.successors : g.neighbors).bind(g);
            const acc = [];
            const visited = {};
            for (const v of vs) {
                if (!g.hasNode(v)) {
                    throw new Error('Graph does not have node: ' + v);
                }
                doDfs(g, v, order === 'post', visited, navigation, acc);
            }
            return acc;
        };
        const postorder = (g, vs) => {
            return dfs(g, vs, 'post');
        };
        const preorder = (g, vs) => {
            return dfs(g, vs, 'pre');
        };

        const removeSelfEdges = (g) => {
            for (const e of g.edges()) {
                if (e.v === e.w) {
                    const node = g.node(e.v);
                    if (!node.selfEdges) {
                        node.selfEdges = [];
                    }
                    node.selfEdges.push({ e: e, label: g.edge(e) });
                    g.removeEdge(e);
                }
            }
        };

        const acyclic_run = (g) => {
            const dfsFAS = (g) => {
                const fas = [];
                const stack = new Set();
                const visited = new Set();
                function dfs(v) {
                    if (visited.has(v)) {
                        return;
                    }
                    visited.add(v);
                    stack.add(v);
                    for (const e of g.outEdges(v)) {
                        if (stack.has(e.w)) {
                            fas.push(e);
                        }
                        else {
                            dfs(e.w);
                        }
                    }
                    stack.delete(v);
                }
                for (const v of g.nodes()) {
                    dfs(v);
                }
                return fas;
            };
            function greedyFAS(g, weightFn) {
                const assignBucket = (buckets, zeroIdx, entry) => {
                    if (!entry.out) {
                        buckets[0].enqueue(entry);
                    }
                    else if (!entry['in']) {
                        buckets[buckets.length - 1].enqueue(entry);
                    }
                    else {
                        buckets[entry.out - entry['in'] + zeroIdx].enqueue(entry);
                    }
                };
                const buildState = (g, weightFn) => {
                    const fasGraph = new dagre.Graph();
                    let maxIn = 0;
                    let maxOut = 0;
                    for (const v of g.nodes()) {
                        fasGraph.setNode(v, { v: v, 'in': 0, out: 0 });
                    }
                    // Aggregate weights on nodes, but also sum the weights across multi-edges into a single edge for the fasGraph.
                    for (const e of g.edges()) {
                        const prevWeight = fasGraph.edge(e.v, e.w) || 0;
                        const weight = weightFn(e);
                        const edgeWeight = prevWeight + weight;
                        fasGraph.setEdge(e.v, e.w, edgeWeight);
                        maxOut = Math.max(maxOut, fasGraph.node(e.v).out += weight);
                        maxIn  = Math.max(maxIn,  fasGraph.node(e.w)['in']  += weight);
                    }
                    const buckets = Array.from(new Array(maxOut + maxIn + 3), () => []);
                    const zeroIdx = maxIn + 1;
                    for (const v of fasGraph.nodes()) {
                        assignBucket(buckets, zeroIdx, fasGraph.node(v));
                    }
                    return { graph: fasGraph, buckets: buckets, zeroIdx: zeroIdx };
                };
                const doGreedyFAS = (g, buckets, zeroIdx) => {
                    const removeNode = (g, buckets, zeroIdx, entry, collectPredecessors) => {
                        const results = collectPredecessors ? [] : undefined;
                        for (const edge of g.inEdges(entry.v)) {
                            const weight = g.edge(edge);
                            const uEntry = g.node(edge.v);
                            if (collectPredecessors) {
                                results.push({ v: edge.v, w: edge.w });
                            }
                            uEntry.out -= weight;
                            assignBucket(buckets, zeroIdx, uEntry);
                        }
                        for (const edge of g.outEdges(entry.v)) {
                            const weight = g.edge(edge);
                            const w = edge.w;
                            const wEntry = g.node(w);
                            wEntry['in'] -= weight;
                            assignBucket(buckets, zeroIdx, wEntry);
                        }
                        g.removeNode(entry.v);
                        return results;
                    };
                    const sources = buckets[buckets.length - 1];
                    const sinks = buckets[0];
                    let results = [];
                    let entry;
                    while (g.nodeCount()) {
                        while ((entry = sinks.dequeue()))   { removeNode(g, buckets, zeroIdx, entry); }
                        while ((entry = sources.dequeue())) { removeNode(g, buckets, zeroIdx, entry); }
                        if (g.nodeCount()) {
                            for (let i = buckets.length - 2; i > 0; --i) {
                                entry = buckets[i].dequeue();
                                if (entry) {
                                    results = results.concat(removeNode(g, buckets, zeroIdx, entry, true));
                                    break;
                                }
                            }
                        }
                    }
                    return results;
                };
                if (g.nodeCount() <= 1) {
                    return [];
                }
                const DEFAULT_WEIGHT_FN = () => 1;
                const state = buildState(g, weightFn || DEFAULT_WEIGHT_FN);
                const results = doGreedyFAS(state.graph, state.buckets, state.zeroIdx);
                // Expand multi-edges
                return flat(results.map((e) => g.outEdges(e.v, e.w)));
            }
            const fas = (g.graph().acyclicer === 'greedy' ? greedyFAS(g, weightFn(g)) : dfsFAS(g));
            for (const e of fas) {
                const label = g.edge(e);
                g.removeEdge(e);
                label.forwardName = e.name;
                label.reversed = true;
                g.setEdge(e.w, e.v, label, uniqueId('rev'));
            }
            function weightFn(g) {
                return function(e) {
                    return g.edge(e).weight;
                };
            }
        };
        const acyclic_undo = (g) => {
            for (const e of g.edges()) {
                const label = g.edge(e);
                if (label.reversed) {
                    g.removeEdge(e);
                    const forwardName = label.forwardName;
                    delete label.reversed;
                    delete label.forwardName;
                    g.setEdge(e.w, e.v, label, forwardName);
                }
            }
        };

        // Returns the amount of slack for the given edge. The slack is defined as the
        // difference between the length of the edge and its minimum length.
        const slack = (g, e) => {
            return g.node(e.w).rank - g.node(e.v).rank - g.edge(e).minlen;
        };

        /*
        * Assigns a rank to each node in the input graph that respects the 'minlen'
        * constraint specified on edges between nodes.
        *
        * This basic structure is derived from Gansner, et al., 'A Technique for
        * Drawing Directed Graphs.'
        *
        * Pre-conditions:
        *
        *    1. Graph must be a connected DAG
        *    2. Graph nodes must be objects
        *    3. Graph edges must have 'weight' and 'minlen' attributes
        *
        * Post-conditions:
        *
        *    1. Graph nodes will have a 'rank' attribute based on the results of the
        *       algorithm. Ranks can start at any index (including negative), we'll
        *       fix them up later.
        */
        const rank = (g) => {
            /*
            * Constructs a spanning tree with tight edges and adjusted the input node's
            * ranks to achieve this. A tight edge is one that is has a length that matches
            * its 'minlen' attribute.
            *
            * The basic structure for this function is derived from Gansner, et al., 'A
            * Technique for Drawing Directed Graphs.'
            *
            * Pre-conditions:
            *
            *    1. Graph must be a DAG.
            *    2. Graph must be connected.
            *    3. Graph must have at least one node.
            *    5. Graph nodes must have been previously assigned a 'rank' property that
            *       respects the 'minlen' property of incident edges.
            *    6. Graph edges must have a 'minlen' property.
            *
            * Post-conditions:
            *
            *    - Graph nodes will have their rank adjusted to ensure that all edges are
            *      tight.
            *
            * Returns a tree (undirected graph) that is constructed using only 'tight'
            * edges.
            */
            const feasibleTree = (g) => {
                const t = new dagre.Graph({ directed: false });
                // Choose arbitrary node from which to start our tree
                const start = g.nodes()[0];
                const size = g.nodeCount();
                t.setNode(start, {});
                let edge;
                let delta;
                // Finds the edge with the smallest slack that is incident on tree and returns it.
                const findMinSlackEdge = (t, g) => {
                    let minKey = Number.POSITIVE_INFINITY;
                    let minValue = undefined;
                    for (const e of g.edges()) {
                        if (t.hasNode(e.v) !== t.hasNode(e.w)) {
                            const key = slack(g, e);
                            if (key < minKey) {
                                minKey = key;
                                minValue = e;
                            }
                        }
                    }
                    return minValue;
                };
                // Finds a maximal tree of tight edges and returns the number of nodes in the tree.
                const tightTree = (t, g) => {
                    const stack = t.nodes().reverse();
                    while (stack.length > 0) {
                        const v = stack.pop();
                        for (const e of g.nodeEdges(v)) {
                            const edgeV = e.v;
                            const w = (v === edgeV) ? e.w : edgeV;
                            if (!t.hasNode(w) && !slack(g, e)) {
                                t.setNode(w, {});
                                t.setEdge(v, w, {});
                                stack.push(w);
                            }
                        }
                    }
                    return t.nodeCount();
                };
                while (tightTree(t, g) < size) {
                    edge = findMinSlackEdge(t, g);
                    delta = t.hasNode(edge.v) ? slack(g, edge) : -slack(g, edge);
                    for (const v of t.nodes()) {
                        g.node(v).rank += delta;
                    }
                }
                return t;
            };
            /*
            * Initializes ranks for the input graph using the longest path algorithm. This
            * algorithm scales well and is fast in practice, it yields rather poor
            * solutions. Nodes are pushed to the lowest layer possible, leaving the bottom
            * ranks wide and leaving edges longer than necessary. However, due to its
            * speed, this algorithm is good for getting an initial ranking that can be fed
            * into other algorithms.
            *
            * This algorithm does not normalize layers because it will be used by other
            * algorithms in most cases. If using this algorithm directly, be sure to
            * run normalize at the end.
            *
            * Pre-conditions:
            *
            *    1. Input graph is a DAG.
            *    2. Input graph node labels can be assigned properties.
            *
            * Post-conditions:
            *
            *    1. Each node will be assign an (unnormalized) 'rank' property.
            */
            const longestPath = (g) => {
                const visited = {};
                const dfs = (v) => {
                    const label = g.node(v);
                    if (visited[v] === true) {
                        return label.rank;
                    }
                    visited[v] = true;
                    let rank = Number.POSITIVE_INFINITY;
                    for (const e of g.outEdges(v)) {
                        const x = dfs(e.w) - g.edge(e).minlen;
                        if (x < rank) {
                            rank = x;
                        }
                    }
                    if (rank === Number.POSITIVE_INFINITY || // return value of _.map([]) for Lodash 3
                        rank === undefined || // return value of _.map([]) for Lodash 4
                        rank === null) { // return value of _.map([null])
                        rank = 0;
                    }
                    return (label.rank = rank);
                };
                for (const v of g.sources()) {
                    dfs(v);
                }
            };
            /*
            * The network simplex algorithm assigns ranks to each node in the input graph
            * and iteratively improves the ranking to reduce the length of edges.
            *
            * Preconditions:
            *
            *    1. The input graph must be a DAG.
            *    2. All nodes in the graph must have an object value.
            *    3. All edges in the graph must have 'minlen' and 'weight' attributes.
            *
            * Postconditions:
            *
            *    1. All nodes in the graph will have an assigned 'rank' attribute that has
            *       been optimized by the network simplex algorithm. Ranks start at 0.
            *
            *
            * A rough sketch of the algorithm is as follows:
            *
            *    1. Assign initial ranks to each node. We use the longest path algorithm,
            *       which assigns ranks to the lowest position possible. In general this
            *       leads to very wide bottom ranks and unnecessarily long edges.
            *    2. Construct a feasible tight tree. A tight tree is one such that all
            *       edges in the tree have no slack (difference between length of edge
            *       and minlen for the edge). This by itself greatly improves the assigned
            *       rankings by shorting edges.
            *    3. Iteratively find edges that have negative cut values. Generally a
            *       negative cut value indicates that the edge could be removed and a new
            *       tree edge could be added to produce a more compact graph.
            *
            * Much of the algorithms here are derived from Gansner, et al., 'A Technique
            * for Drawing Directed Graphs.' The structure of the file roughly follows the
            * structure of the overall algorithm.
            */
            const networkSimplex = (g) => {
                /*
                * Returns a new graph with only simple edges. Handles aggregation of data
                * associated with multi-edges.
                */
                const simplify = (g) => {
                    const graph = new dagre.Graph();
                    graph.setGraph(g.graph());
                    for (const v of g.nodes()) {
                        graph.setNode(v, g.node(v));
                    }
                    for (const e of g.edges()) {
                        const simpleLabel = graph.edge(e.v, e.w) || { weight: 0, minlen: 1 };
                        const label = g.edge(e);
                        graph.setEdge(e.v, e.w, {
                            weight: simpleLabel.weight + label.weight,
                            minlen: Math.max(simpleLabel.minlen, label.minlen)
                        });
                    }
                    return graph;
                };
                g = simplify(g);
                longestPath(g);
                const tree = feasibleTree(g);
                const initLowLimValues = (tree, root) => {
                    root = tree.nodes()[0];
                    const dfsAssignLowLim = (tree, visited, nextLim, v, parent) => {
                        const low = nextLim;
                        const label = tree.node(v);
                        visited[v] = true;
                        for (const w of tree.neighbors(v)) {
                            if (!(w in visited)) {
                                nextLim = dfsAssignLowLim(tree, visited, nextLim, w, v);
                            }
                        }
                        label.low = low;
                        label.lim = nextLim++;
                        if (parent) {
                            label.parent = parent;
                        }
                        else {
                            // TODO should be able to remove this when we incrementally update low lim
                            delete label.parent;
                        }
                        return nextLim;
                    };
                    dfsAssignLowLim(tree, {}, 1, root);
                };
                initLowLimValues(tree);
                // Initializes cut values for all edges in the tree.
                const initCutValues = (t, g) => {
                    // Given the tight tree, its graph, and a child in the graph calculate and
                    // return the cut value for the edge between the child and its parent.
                    const calcCutValue = (t, g, child) => {
                        const childLab = t.node(child);
                        const parent = childLab.parent;
                        // True if the child is on the tail end of the edge in the directed graph
                        let childIsTail = true;
                        // The graph's view of the tree edge we're inspecting
                        let graphEdge = g.edge(child, parent);
                        // The accumulated cut value for the edge between this node and its parent
                        if (!graphEdge) {
                            childIsTail = false;
                            graphEdge = g.edge(parent, child);
                        }
                        let cutValue = graphEdge.weight;
                        for (const e of g.nodeEdges(child)) {
                            const isOutEdge = e.v === child;
                            const other = isOutEdge ? e.w : e.v;
                            if (other !== parent) {
                                const pointsToHead = isOutEdge === childIsTail;
                                const otherWeight = g.edge(e).weight;
                                cutValue += pointsToHead ? otherWeight : -otherWeight;
                                if (t.hasEdge(child, other)) {
                                    const otherCutValue = t.edge(child, other).cutvalue;
                                    cutValue += pointsToHead ? -otherCutValue : otherCutValue;
                                }
                            }
                        }
                        return cutValue;
                    };
                    const assignCutValue = (t, g, child) => {
                        const childLab = t.node(child);
                        const parent = childLab.parent;
                        t.edge(child, parent).cutvalue = calcCutValue(t, g, child);
                    };
                    let vs = postorder(t, t.nodes());
                    vs = vs.slice(0, vs.length - 1);
                    for (const v of vs) {
                        assignCutValue(t, g, v);
                    }
                };
                initCutValues(tree, g);
                const leaveEdge = (tree) => {
                    return tree.edges().find((e) => tree.edge(e).cutvalue < 0);
                };
                const enterEdge = (t, g, edge) => {
                    let v = edge.v;
                    let w = edge.w;
                    // For the rest of this function we assume that v is the tail and w is the
                    // head, so if we don't have this edge in the graph we should flip it to
                    // match the correct orientation.
                    if (!g.hasEdge(v, w)) {
                        v = edge.w;
                        w = edge.v;
                    }
                    const vLabel = t.node(v);
                    const wLabel = t.node(w);
                    let tailLabel = vLabel;
                    let flip = false;
                    // If the root is in the tail of the edge then we need to flip the logic that
                    // checks for the head and tail nodes in the candidates function below.
                    if (vLabel.lim > wLabel.lim) {
                        tailLabel = wLabel;
                        flip = true;
                    }
                    // Returns true if the specified node is descendant of the root node per the
                    // assigned low and lim attributes in the tree.
                    const isDescendant = (tree, vLabel, rootLabel) => {
                        return rootLabel.low <= vLabel.lim && vLabel.lim <= rootLabel.lim;
                    };
                    const candidates = g.edges().filter((edge) => flip === isDescendant(t, t.node(edge.v), tailLabel) && flip !== isDescendant(t, t.node(edge.w), tailLabel));
                    let minKey = Number.POSITIVE_INFINITY;
                    let minValue = undefined;
                    for (const edge of candidates) {
                        const key = slack(g, edge);
                        if (key < minKey) {
                            minKey = key;
                            minValue = edge;
                        }
                    }
                    return minValue;
                };
                const exchangeEdges = (t, g, e, f) => {
                    const v = e.v;
                    const w = e.w;
                    t.removeEdge(v, w);
                    t.setEdge(f.v, f.w, {});
                    initLowLimValues(t);
                    initCutValues(t, g);
                    const updateRanks = (t, g) => {
                        const root = t.nodes().find((v) => !g.node(v).parent);
                        let vs = preorder(t, root);
                        vs = vs.slice(1);
                        for (const v of vs) {
                            const parent = t.node(v).parent;
                            let edge = g.edge(v, parent);
                            let flipped = false;
                            if (!edge) {
                                edge = g.edge(parent, v);
                                flipped = true;
                            }
                            g.node(v).rank = g.node(parent).rank + (flipped ? edge.minlen : -edge.minlen);
                        }
                    };
                    updateRanks(t, g);
                };
                let e;
                let f;
                while ((e = leaveEdge(tree))) {
                    f = enterEdge(tree, g, e);
                    exchangeEdges(tree, g, e, f);
                }
            };

            switch(g.graph().ranker) {
                case 'tight-tree': {
                    longestPath(g);
                    feasibleTree(g);
                    break;
                }
                case 'longest-path': {
                    longestPath(g);
                    break;
                }
                default: {
                    networkSimplex(g);
                    break;
                }
            }
        };

        /*
        * Creates temporary dummy nodes that capture the rank in which each edge's
        * label is going to, if it has one of non-zero width and height. We do this
        * so that we can safely remove empty ranks while preserving balance for the
        * label's position.
        */
        const injectEdgeLabelProxies = (g) => {
            for (const e of g.edges()) {
                const edge = g.edge(e);
                if (edge.width && edge.height) {
                    const v = g.node(e.v);
                    const w = g.node(e.w);
                    const label = { rank: (w.rank - v.rank) / 2 + v.rank, e: e };
                    util_addDummyNode(g, 'edge-proxy', label, '_ep');
                }
            }
        };

        const removeEmptyRanks = (g) => {
            // Ranks may not start at 0, so we need to offset them
            const layers = [];
            if (g.nodes().length > 0) {
                const ranks = g.nodes().map((v) => g.node(v).rank).filter((rank) => rank != undefined);
                const offset = Math.min(...ranks);
                for (const v of g.nodes()) {
                    const rank = g.node(v).rank - offset;
                    if (!layers[rank]) {
                        layers[rank] = [];
                    }
                    layers[rank].push(v);
                }
            }
            let delta = 0;
            const nodeRankFactor = g.graph().nodeRankFactor;
            for (let i = 0; i < layers.length; i++) {
                const vs = layers[i];
                if (vs === undefined && i % nodeRankFactor !== 0) {
                    --delta;
                }
                else if (delta && vs) {
                    for (const v of vs) {
                        g.node(v).rank += delta;
                    }
                }
            }
        };

        /*
        * A nesting graph creates dummy nodes for the tops and bottoms of subgraphs,
        * adds appropriate edges to ensure that all cluster nodes are placed between
        * these boundries, and ensures that the graph is connected.
        *
        * In addition we ensure, through the use of the minlen property, that nodes
        * and subgraph border nodes to not end up on the same rank.
        *
        * Preconditions:
        *
        *    1. Input graph is a DAG
        *    2. Nodes in the input graph has a minlen attribute
        *
        * Postconditions:
        *
        *    1. Input graph is connected.
        *    2. Dummy nodes are added for the tops and bottoms of subgraphs.
        *    3. The minlen attribute for nodes is adjusted to ensure nodes do not
        *       get placed on the same rank as subgraph border nodes.
        *
        * The nesting graph idea comes from Sander, 'Layout of Compound Directed
        * Graphs.'
        */
        const nestingGraph_run = (g) => {
            const root = util_addDummyNode(g, 'root', {}, '_root');
            const treeDepths = (g) => {
                const depths = {};
                const dfs = (v, depth) => {
                    const children = g.children(v);
                    if (children && children.length) {
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
                const top = util_addDummyNode(g, 'border', { width: 0, height: 0 }, '_bt');
                const bottom = util_addDummyNode(g, 'border', { width: 0, height: 0 }, '_bb');
                const label = g.node(v);
                g.setParent(top, v);
                label.borderTop = top;
                g.setParent(bottom, v);
                label.borderBottom = bottom;
                for (const child of children) {
                    dfs(g, root, nodeSep, weight, height, depths, child);
                    const childNode = g.node(child);
                    const childTop = childNode.borderTop ? childNode.borderTop : child;
                    const childBottom = childNode.borderBottom ? childNode.borderBottom : child;
                    const thisWeight = childNode.borderTop ? weight : 2 * weight;
                    const minlen = childTop !== childBottom ? 1 : height - depths[v] + 1;
                    g.setEdge(top, childTop, {
                        weight: thisWeight,
                        minlen: minlen,
                        nestingEdge: true
                    });
                    g.setEdge(childBottom, bottom, {
                        weight: thisWeight,
                        minlen: minlen,
                        nestingEdge: true
                    });
                }
                if (!g.parent(v)) {
                    g.setEdge(root, top, { weight: 0, minlen: height + depths[v] });
                }
            };
            const depths = treeDepths(g);
            const height = Math.max(...Object.values(depths)) - 1; // Note: depths is an Object not an array
            const nodeSep = 2 * height + 1;
            g.graph().nestingRoot = root;
            // Multiply minlen by nodeSep to align nodes on non-border ranks.
            for (const e of g.edges()) {
                g.edge(e).minlen *= nodeSep;
            }
            // Calculate a weight that is sufficient to keep subgraphs vertically compact
            const sumWeights = (g) => {
                return g.edges().reduce((acc, e) => acc + g.edge(e).weight, 0);
            };
            const weight = sumWeights(g) + 1;
            // Create border nodes and link them up
            for (const child of g.children()) {
                dfs(g, root, nodeSep, weight, height, depths, child);
            }
            // Save the multiplier for node layers for later removal of empty border layers.
            g.graph().nodeRankFactor = nodeSep;
        };
        const nestingGraph_cleanup = (g) => {
            const graphLabel = g.graph();
            g.removeNode(graphLabel.nestingRoot);
            delete graphLabel.nestingRoot;
            for (const e of g.edges()) {
                const edge = g.edge(e);
                if (edge.nestingEdge) {
                    g.removeEdge(e);
                }
            }
        };

        // Adjusts the ranks for all nodes in the graph such that all nodes v have rank(v) >= 0 and at least one node w has rank(w) = 0.
        const normalizeRanks = (g) => {
            let min = Number.POSITIVE_INFINITY;
            for (const v of g.nodes()) {
                const rank = g.node(v).rank;
                if (rank !== undefined && rank < min) {
                    min = rank;
                }
            }
            for (const v of g.nodes()) {
                const node = g.node(v);
                if (node.rank !== undefined) {
                    node.rank -= min;
                }
            }
        };

        const assignRankMinMax = (g) => {
            let maxRank = 0;
            for (const v of g.nodes()) {
                const node = g.node(v);
                if (node.borderTop) {
                    node.minRank = g.node(node.borderTop).rank;
                    node.maxRank = g.node(node.borderBottom).rank;
                    maxRank = Math.max(maxRank, node.maxRank);
                }
            }
            g.graph().maxRank = maxRank;
        };

        /*
        * Breaks any long edges in the graph into short segments that span 1 layer
        * each. This operation is undoable with the denormalize function.
        *
        * Pre-conditions:
        *
        *    1. The input graph is a DAG.
        *    2. Each node in the graph has a 'rank' property.
        *
        * Post-condition:
        *
        *    1. All edges in the graph have a length of 1.
        *    2. Dummy nodes are added where edges have been split into segments.
        *    3. The graph is augmented with a 'dummyChains' attribute which contains
        *       the first dummy in each chain of dummy nodes produced.
        */
        const normalize = (g) => {
            g.graph().dummyChains = [];
            for (const e of g.edges()) {
                let v = e.v;
                let vRank = g.node(v).rank;
                const w = e.w;
                const wRank = g.node(w).rank;
                const name = e.name;
                const edgeLabel = g.edge(e);
                const labelRank = edgeLabel.labelRank;
                if (wRank !== vRank + 1) {
                    g.removeEdge(e);
                    let dummy;
                    let attrs;
                    let i;
                    for (i = 0, ++vRank; vRank < wRank; ++i, ++vRank) {
                        edgeLabel.points = [];
                        attrs = {
                            width: 0, height: 0,
                            edgeLabel: edgeLabel, edgeObj: e,
                            rank: vRank
                        };
                        dummy = util_addDummyNode(g, 'edge', attrs, '_d');
                        if (vRank === labelRank) {
                            attrs.width = edgeLabel.width;
                            attrs.height = edgeLabel.height;
                            attrs.dummy = 'edge-label';
                            attrs.labelpos = edgeLabel.labelpos;
                        }
                        g.setEdge(v, dummy, { weight: edgeLabel.weight }, name);
                        if (i === 0) {
                            g.graph().dummyChains.push(dummy);
                        }
                        v = dummy;
                    }
                    g.setEdge(v, w, { weight: edgeLabel.weight }, name);
                }
            }
        };

        const denormalize = (g) => {
            const dummyChains = g.graph().dummyChains;
            dummyChains.forEach(function(v) {
                let node = g.node(v);
                const origLabel = node.edgeLabel;
                let w;
                g.setEdge(node.edgeObj, origLabel);
                while (node.dummy) {
                    w = g.successors(v)[0];
                    g.removeNode(v);
                    origLabel.points.push({ x: node.x, y: node.y });
                    if (node.dummy === 'edge-label') {
                        origLabel.x = node.x;
                        origLabel.y = node.y;
                        origLabel.width = node.width;
                        origLabel.height = node.height;
                    }
                    v = w;
                    node = g.node(v);
                }
            });
        };

        const removeEdgeLabelProxies = (g) => {
            for (const v of g.nodes()) {
                const node = g.node(v);
                if (node.dummy === 'edge-proxy') {
                    g.edge(node.e).labelRank = node.rank;
                    g.removeNode(v);
                }
            }
        };

        const parentDummyChains = (g) => {
            // Find a path from v to w through the lowest common ancestor (LCA). Return the full path and the LCA.
            const findPath = (g, postorderNums, v, w) => {
                const vPath = [];
                const wPath = [];
                const low = Math.min(postorderNums[v].low, postorderNums[w].low);
                const lim = Math.max(postorderNums[v].lim, postorderNums[w].lim);
                // Traverse up from v to find the LCA
                let parent = v;
                do {
                    parent = g.parent(parent);
                    vPath.push(parent);
                }
                while (parent && (postorderNums[parent].low > low || lim > postorderNums[parent].lim));
                const lca = parent;
                // Traverse from w to LCA
                parent = w;
                while ((parent = g.parent(parent)) !== lca) {
                    wPath.push(parent);
                }
                return { path: vPath.concat(wPath.reverse()), lca: lca };
            };
            const postorder = (g) => {
                const result = {};
                let lim = 0;
                const dfs = (v) => {
                    const low = lim;
                    for (const u of g.children(v)) {
                        dfs(u);
                    }
                    result[v] = { low: low, lim: lim++ };
                };
                for (const v of g.children()) {
                    dfs(v);
                }
                return result;
            };
            const postorderNums = postorder(g);
            const dummyChains = g.graph().dummyChains;
            if (dummyChains) {
                dummyChains.forEach(function(v) {
                    let node = g.node(v);
                    const edgeObj = node.edgeObj;
                    const pathData = findPath(g, postorderNums, edgeObj.v, edgeObj.w);
                    const path = pathData.path;
                    const lca = pathData.lca;
                    let pathIdx = 0;
                    let pathV = path[pathIdx];
                    let ascending = true;
                    while (v !== edgeObj.w) {
                        node = g.node(v);
                        if (ascending) {
                            while ((pathV = path[pathIdx]) !== lca && g.node(pathV).maxRank < node.rank) {
                                pathIdx++;
                            }
                            if (pathV === lca) {
                                ascending = false;
                            }
                        }
                        if (!ascending) {
                            while (pathIdx < path.length - 1 && g.node(pathV = path[pathIdx + 1]).minRank <= node.rank) {
                                pathIdx++;
                            }
                            pathV = path[pathIdx];
                        }
                        g.setParent(v, pathV);
                        v = g.successors(v)[0];
                    }
                });
            }
        };

        const addBorderSegments = (g) => {
            const addBorderNode = (g, prop, prefix, sg, sgNode, rank) => {
                const label = { width: 0, height: 0, rank: rank, borderType: prop };
                const prev = sgNode[prop][rank - 1];
                const curr = util_addDummyNode(g, 'border', label, prefix);
                sgNode[prop][rank] = curr;
                g.setParent(curr, sg);
                if (prev) {
                    g.setEdge(prev, curr, { weight: 1 });
                }
            };
            const dfs = (v) => {
                const children = g.children(v);
                const node = g.node(v);
                if (children.length) {
                    for (const v of children) {
                        dfs(v);
                    }
                }
                if ('minRank' in node) {
                    node.borderLeft = [];
                    node.borderRight = [];
                    for (let rank = node.minRank, maxRank = node.maxRank + 1; rank < maxRank; ++rank) {
                        addBorderNode(g, 'borderLeft', '_bl', v, node, rank);
                        addBorderNode(g, 'borderRight', '_br', v, node, rank);
                    }
                }
            };
            for (const v of g.children()) {
                dfs(v);
            }
        };

        /*
        * Applies heuristics to minimize edge crossings in the graph and sets the best
        * order solution as an order attribute on each node.
        *
        * Pre-conditions:
        *
        *    1. Graph must be DAG
        *    2. Graph nodes must be objects with a 'rank' attribute
        *    3. Graph edges must have the 'weight' attribute
        *
        * Post-conditions:
        *
        *    1. Graph nodes will have an 'order' attribute based on the results of the algorithm.
        */
        const order = (g) => {
            const sortSubgraph = (g, v, cg, biasRight) => {
                /*
                * Given a list of entries of the form {v, barycenter, weight} and a
                * constraint graph this function will resolve any conflicts between the
                * constraint graph and the barycenters for the entries. If the barycenters for
                * an entry would violate a constraint in the constraint graph then we coalesce
                * the nodes in the conflict into a new node that respects the contraint and
                * aggregates barycenter and weight information.
                *
                * This implementation is based on the description in Forster, 'A Fast and Simple Hueristic for Constrained Two-Level Crossing Reduction,' thought it differs in some specific details.
                *
                * Pre-conditions:
                *
                *    1. Each entry has the form {v, barycenter, weight}, or if the node has
                *       no barycenter, then {v}.
                *
                * Returns:
                *
                *    A new list of entries of the form {vs, i, barycenter, weight}. The list
                *    `vs` may either be a singleton or it may be an aggregation of nodes
                *    ordered such that they do not violate constraints from the constraint
                *    graph. The property `i` is the lowest original index of any of the
                *    elements in `vs`.
                */
                const resolveConflicts = (entries, cg) => {
                    const mergeEntries = (target, source) => {
                        let sum = 0;
                        let weight = 0;
                        if (target.weight) {
                            sum += target.barycenter * target.weight;
                            weight += target.weight;
                        }
                        if (source.weight) {
                            sum += source.barycenter * source.weight;
                            weight += source.weight;
                        }
                        target.vs = source.vs.concat(target.vs);
                        target.barycenter = sum / weight;
                        target.weight = weight;
                        target.i = Math.min(source.i, target.i);
                        source.merged = true;
                    };
                    const mappedEntries = {};
                    entries.forEach(function(entry, i) {
                        const tmp = mappedEntries[entry.v] = {
                            indegree: 0,
                            'in': [],
                            out: [],
                            vs: [entry.v],
                            i: i
                        };
                        if (entry.barycenter !== undefined) {
                            tmp.barycenter = entry.barycenter;
                            tmp.weight = entry.weight;
                        }
                    });
                    for (const e of cg.edges()) {
                        const entryV = mappedEntries[e.v];
                        const entryW = mappedEntries[e.w];
                        if (entryV !== undefined && entryW !== undefined) {
                            entryW.indegree++;
                            entryV.out.push(mappedEntries[e.w]);
                        }
                    }
                    const sourceSet = Object.values(mappedEntries).filter((entry) => !entry.indegree);
                    const results = [];
                    function handleIn(vEntry) {
                        return function(uEntry) {
                            if (uEntry.merged) {
                                return;
                            }
                            if (uEntry.barycenter === undefined || vEntry.barycenter === undefined || uEntry.barycenter >= vEntry.barycenter) {
                                mergeEntries(vEntry, uEntry);
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
                    const pick = (obj, attrs) => {
                        const value = {};
                        for (const key of attrs) {
                            if (obj[key] !== undefined) {
                                value[key] = obj[key];
                            }
                        }
                        return value;
                    };
                    return Object.values(results).filter((entry) => !entry.merged).map((entry) => pick(entry, ['vs', 'i', 'barycenter', 'weight']));
                };
                let movable = g.children(v);
                const node = g.node(v);
                const bl = node ? node.borderLeft : undefined;
                const br = node ? node.borderRight: undefined;
                const subgraphs = {};
                if (bl) {
                    movable = movable.filter((w) => w !== bl && w !== br);
                }
                const barycenter = (g, movable) => {
                    return (movable || []).map((v) => {
                        const inV = g.inEdges(v);
                        if (!inV.length) {
                            return { v: v };
                        }
                        else {
                            const result = inV.reduce((acc, e) => {
                                const edge = g.edge(e);
                                const nodeU = g.node(e.v);
                                return {
                                    sum: acc.sum + (edge.weight * nodeU.order),
                                    weight: acc.weight + edge.weight
                                };
                            }, { sum: 0, weight: 0 });
                            return {
                                v: v,
                                barycenter: result.sum / result.weight,
                                weight: result.weight
                            };
                        }
                    });
                };
                const mergeBarycenters = (target, other) => {
                    if (target.barycenter !== undefined) {
                        target.barycenter = (target.barycenter * target.weight + other.barycenter * other.weight) / (target.weight + other.weight);
                        target.weight += other.weight;
                    }
                    else {
                        target.barycenter = other.barycenter;
                        target.weight = other.weight;
                    }
                };
                const barycenters = barycenter(g, movable);
                for (const entry of barycenters) {
                    if (g.children(entry.v).length) {
                        const subgraphResult = sortSubgraph(g, entry.v, cg, biasRight);
                        subgraphs[entry.v] = subgraphResult;
                        if ('barycenter' in subgraphResult) {
                            mergeBarycenters(entry, subgraphResult);
                        }
                    }
                }
                const entries = resolveConflicts(barycenters, cg);
                // expand subgraphs
                for (const entry of entries) {
                    entry.vs = flat(entry.vs.map((v) => subgraphs[v] ? subgraphs[v].vs : v));
                }
                const sort = (entries, biasRight) => {
                    const consumeUnsortable = (vs, unsortable, index) => {
                        let last;
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
                            }
                            else if (entryV.barycenter > entryW.barycenter) {
                                return 1;
                            }
                            return !bias ? entryV.i - entryW.i : entryW.i - entryV.i;
                        };
                    };
                    // partition
                    const parts = { lhs: [], rhs: [] };
                    for (const value of entries) {
                        if ('barycenter' in value) {
                            parts.lhs.push(value);
                        }
                        else {
                            parts.rhs.push(value);
                        }
                    }
                    const sortable = parts.lhs;
                    const unsortable = parts.rhs.sort((a, b) => -a.i + b.i);
                    const vs = [];
                    let sum = 0;
                    let weight = 0;
                    let vsIndex = 0;
                    sortable.sort(compareWithBias(!!biasRight));
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
                const result = sort(entries, biasRight);
                if (bl) {
                    result.vs = flat([bl, result.vs, br]);
                    if (g.predecessors(bl).length) {
                        const blPred = g.node(g.predecessors(bl)[0]);
                        const brPred = g.node(g.predecessors(br)[0]);
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
            const addSubgraphConstraints = (g, cg, vs) => {
                const prev = {};
                let rootPrev;
                for (const v of vs) {
                    let child = g.parent(v);
                    let parent;
                    let prevChild;
                    while (child) {
                        parent = g.parent(child);
                        if (parent) {
                            prevChild = prev[parent];
                            prev[parent] = child;
                        }
                        else {
                            prevChild = rootPrev;
                            rootPrev = child;
                        }
                        if (prevChild && prevChild !== child) {
                            cg.setEdge(prevChild, child);
                            return;
                        }
                        child = parent;
                    }
                }
            };
            const sweepLayerGraphs = (layerGraphs, biasRight) => {
                const cg = new dagre.Graph();
                for (const lg of layerGraphs) {
                    const root = lg.graph().root;
                    const sorted = sortSubgraph(lg, root, cg, biasRight);
                    const vs = sorted.vs;
                    const length = vs.length;
                    for (let i = 0; i < length; i++) {
                        lg.node(vs[i]).order = i;
                    }
                    addSubgraphConstraints(lg, cg, sorted.vs);
                }
            };
            const twoLayerCrossCount = (g, northLayer, southLayer) => {
                // Sort all of the edges between the north and south layers by their position
                // in the north layer and then the south. Map these edges to the position of
                // their head in the south layer.
                const southPos = {};
                for (let i = 0; i < southLayer.length; i++) {
                    southPos[southLayer[i]] = i;
                }
                const southEntries = flat(northLayer.map((v) => g.outEdges(v).map((e) => { return { pos: southPos[e.w], weight: g.edge(e).weight }; }).sort((a, b) => a.pos - b.pos)));
                // Build the accumulator tree
                let firstIndex = 1;
                while (firstIndex < southLayer.length) {
                    firstIndex <<= 1;
                }
                const tree = Array.from(new Array(2 * firstIndex - 1), () => 0);
                firstIndex -= 1;
                // Calculate the weighted crossings
                let cc = 0;
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
                    cc += entry.weight * weightSum;
                }
                return cc;
            };
            /*
            * A function that takes a layering (an array of layers, each with an array of
            * ordererd nodes) and a graph and returns a weighted crossing count.
            *
            * Pre-conditions:
            *
            *    1. Input graph must be simple (not a multigraph), directed, and include
            *       only simple edges.
            *    2. Edges in the input graph must have assigned weights.
            *
            * Post-conditions:
            *
            *    1. The graph and layering matrix are left unchanged.
            *
            * This algorithm is derived from Barth, et al., 'Bilayer Cross Counting.'
            */
            const crossCount = (g, layering) => {
                let count = 0;
                for (let i = 1; i < layering.length; i++) {
                    count += twoLayerCrossCount(g, layering[i - 1], layering[i]);
                }
                return count;
            };
            /*
            * Assigns an initial order value for each node by performing a DFS search
            * starting from nodes in the first rank. Nodes are assigned an order in their
            * rank as they are first visited.
            *
            * This approach comes from Gansner, et al., 'A Technique for Drawing Directed
            * Graphs.'
            *
            * Returns a layering matrix with an array per layer and each layer sorted by
            * the order of its nodes.
            */
            const initOrder = (g) => {
                const visited = {};
                const nodes = g.nodes().filter((v) => !g.children(v).length);
                let maxRank = undefined;
                for (const v of nodes) {
                    if (!g.children(v).length > 0) {
                        const rank = g.node(v).rank;
                        if (maxRank === undefined || (rank !== undefined && rank > maxRank)) {
                            maxRank = rank;
                        }
                    }
                }
                if (maxRank !== undefined) {
                    const layers = Array.from(new Array(maxRank + 1), () => []);
                    for (const v of nodes.map((v) => [ g.node(v).rank, v ]).sort((a, b) => a[0] - b[0]).map((item) => item[1])) {
                        const queue = [ v ];
                        while (queue.length > 0) {
                            const v = queue.shift();
                            if (visited[v] !== true) {
                                visited[v] = true;
                                const rank = g.node(v).rank;
                                layers[rank].push(v);
                                queue.push(...g.successors(v));
                            }
                        }
                    }
                    return layers;
                }
                return [];
            };
            /*
            * Constructs a graph that can be used to sort a layer of nodes. The graph will
            * contain all base and subgraph nodes from the request layer in their original
            * hierarchy and any edges that are incident on these nodes and are of the type
            * requested by the 'relationship' parameter.
            *
            * Nodes from the requested rank that do not have parents are assigned a root
            * node in the output graph, which is set in the root graph attribute. This
            * makes it easy to walk the hierarchy of movable nodes during ordering.
            *
            * Pre-conditions:
            *
            *    1. Input graph is a DAG
            *    2. Base nodes in the input graph have a rank attribute
            *    3. Subgraph nodes in the input graph has minRank and maxRank attributes
            *    4. Edges have an assigned weight
            *
            * Post-conditions:
            *
            *    1. Output graph has all nodes in the movable rank with preserved
            *       hierarchy.
            *    2. Root nodes in the movable layer are made children of the node
            *       indicated by the root attribute of the graph.
            *    3. Non-movable nodes incident on movable nodes, selected by the
            *       relationship parameter, are included in the graph (without hierarchy).
            *    4. Edges incident on movable nodes, selected by the relationship
            *       parameter, are added to the output graph.
            *    5. The weights for copied edges are aggregated as need, since the output
            *       graph is not a multi-graph.
            */
            const buildLayerGraph = (g, rank, relationship) => {
                let root;
                while (g.hasNode((root = uniqueId('_root'))));
                const graph = new dagre.Graph({ compound: true });
                graph.setGraph({ root: root });
                graph.setDefaultNodeLabel((v) => g.node(v));
                for (const v of g.nodes()) {
                    const node = g.node(v);
                    if (node.rank === rank || node.minRank <= rank && rank <= node.maxRank) {
                        graph.setNode(v);
                        const parent = g.parent(v);
                        graph.setParent(v, parent || root);
                        // This assumes we have only short edges!
                        for (const e of g[relationship](v)) {
                            const u = e.v === v ? e.w : e.v;
                            const edge = graph.edge(u, v);
                            const weight = edge !== undefined ? edge.weight : 0;
                            graph.setEdge(u, v, { weight: g.edge(e).weight + weight });
                        }
                        if ('minRank' in node) {
                            graph.setNode(v, {
                                borderLeft: node.borderLeft[rank],
                                borderRight: node.borderRight[rank]
                            });
                        }
                    }
                }
                return graph;
            };
            const rank = maxRank(g);
            const downLayerGraphs = new Array(rank !== undefined ? rank : 0);
            const upLayerGraphs = new Array(rank !== undefined ? rank : 0);
            for (let i = 0; i < rank; i++) {
                downLayerGraphs[i] = buildLayerGraph(g, i + 1, 'inEdges');
                upLayerGraphs[i] = buildLayerGraph(g, rank - i - 1, 'outEdges');
            }
            let layering = initOrder(g);
            const assignOrder = (g, layering) => {
                for (const layer of layering) {
                    for (let i = 0; i < layer.length; i++) {
                        g.node(layer[i]).order = i;
                    }
                }
            };
            assignOrder(g, layering);
            let bestCC = Number.POSITIVE_INFINITY;
            let best;
            for (let i = 0, lastBest = 0; lastBest < 4; ++i, ++lastBest) {
                sweepLayerGraphs(i % 2 ? downLayerGraphs : upLayerGraphs, i % 4 >= 2);
                layering = buildLayerMatrix(g);
                const cc = crossCount(g, layering);
                if (cc < bestCC) {
                    lastBest = 0;
                    const length = layering.length;
                    best = new Array(length);
                    for (let i = 0; i < length; i++) {
                        best[i] = layering[i].slice();
                    }
                    bestCC = cc;
                }
            }
            assignOrder(g, best);
        };

        const insertSelfEdges = (g) => {
            const layers = buildLayerMatrix(g);
            for (const layer of layers) {
                let orderShift = 0;
                layer.forEach(function(v, i) {
                    const node = g.node(v);
                    node.order = i + orderShift;
                    if (node.selfEdges) {
                        for (const selfEdge of node.selfEdges) {
                            util_addDummyNode(g, 'selfedge', {
                                width: selfEdge.label.width,
                                height: selfEdge.label.height,
                                rank: node.rank,
                                order: i + (++orderShift),
                                e: selfEdge.e,
                                label: selfEdge.label
                            }, '_se');
                        }
                        delete node.selfEdges;
                    }
                });
            }
        };

        const coordinateSystem_adjust = (g) => {
            const rankDir = g.graph().rankdir.toLowerCase();
            if (rankDir === 'lr' || rankDir === 'rl') {
                coordinateSystem_swapWidthHeight(g);
            }
        };

        const coordinateSystem_undo = (g) => {
            const swapXY = (g) => {
                const swapXYOne = (attrs) => {
                    const x = attrs.x;
                    attrs.x = attrs.y;
                    attrs.y = x;
                };
                for (const v of g.nodes()) {
                    swapXYOne(g.node(v));
                }
                for (const e of g.edges()) {
                    const edge = g.edge(e);
                    for (const e of edge.points) {
                        swapXYOne(e);
                    }
                    if (edge.x !== undefined) {
                        swapXYOne(edge);
                    }
                }
            };
            const rankDir = g.graph().rankdir.toLowerCase();
            if (rankDir === 'bt' || rankDir === 'rl') {
                for (const v of g.nodes()) {
                    const attr = g.node(v);
                    attr.y = -attr.y;
                }
                for (const e of g.edges()) {
                    const edge = g.edge(e);
                    for (const attr of edge.points) {
                        attr.y = -attr.y;
                    }
                    if ('y' in edge) {
                        edge.y = -edge.y;
                    }
                }
            }
            if (rankDir === 'lr' || rankDir === 'rl') {
                swapXY(g);
                coordinateSystem_swapWidthHeight(g);
            }
        };
        const coordinateSystem_swapWidthHeight = (g) => {
            const swapWidthHeightOne = (attrs) => {
                const w = attrs.width;
                attrs.width = attrs.height;
                attrs.height = w;
            };
            for (const v of g.nodes()) {
                swapWidthHeightOne(g.node(v));
            }
            for (const e of g.edges()) {
                swapWidthHeightOne(g.edge(e));
            }
        };

        const position = (g) => {
            // Coordinate assignment based on Brandes and Köpf, 'Fast and Simple Horizontal Coordinate Assignment.'
            const positionX = (g) => {
                const findOtherInnerSegmentNode = (g, v) => {
                    if (g.node(v).dummy) {
                        return g.predecessors(v).find((u) => g.node(u).dummy);
                    }
                };
                const addConflict = (conflicts, v, w) => {
                    if (v > w) {
                        const tmp = v;
                        v = w;
                        w = tmp;
                    }
                    let conflictsV = conflicts[v];
                    if (!conflictsV) {
                        conflicts[v] = conflictsV = {};
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
                /*
                * Try to align nodes into vertical 'blocks' where possible. This algorithm
                * attempts to align a node with one of its median neighbors. If the edge
                * connecting a neighbor is a type-1 conflict then we ignore that possibility.
                * If a previous node has already formed a block with a node after the node
                * we're trying to form a block with, we also ignore that possibility - our
                * blocks would be split in that scenario.
                */
                const verticalAlignment = (g, layering, conflicts, neighborFn) => {
                    const root = {};
                    const align = {};
                    const pos = {};
                    // We cache the position here based on the layering because the graph and
                    // layering may be out of sync. The layering matrix is manipulated to
                    // generate different extreme alignments.
                    for (const layer of layering) {
                        layer.forEach(function(v, order) {
                            root[v] = v;
                            align[v] = v;
                            pos[v] = order;
                        });
                    }
                    for (const layer of layering) {
                        let prevIdx = -1;
                        for (const v of layer) {
                            let ws = neighborFn(v);
                            if (ws.length) {
                                ws = ws.sort((a, b) => pos[a] - pos[b]);
                                const mp = (ws.length - 1) / 2;
                                for (let i = Math.floor(mp), il = Math.ceil(mp); i <= il; ++i) {
                                    const w = ws[i];
                                    if (align[v] === v && prevIdx < pos[w] && !hasConflict(conflicts, v, w)) {
                                        align[w] = v;
                                        align[v] = root[v] = root[w];
                                        prevIdx = pos[w];
                                    }
                                }
                            }
                        }
                    }
                    return { root: root, align: align };
                };
                const horizontalCompaction = (g, layering, root, align, reverseSep) => {
                    // This portion of the algorithm differs from BK due to a number of problems.
                    // Instead of their algorithm we construct a new block graph and do two
                    // sweeps. The first sweep places blocks with the smallest possible
                    // coordinates. The second sweep removes unused space by moving blocks to the
                    // greatest coordinates without violating separation.
                    const xs = {};
                    const blockG = buildBlockGraph(g, layering, root, reverseSep);
                    const borderType = reverseSep ? 'borderLeft' : 'borderRight';
                    const iterate = (setXsFunc, nextNodesFunc) => {
                        let stack = blockG.nodes();
                        let elem = stack.pop();
                        const visited = {};
                        while (elem) {
                            if (visited[elem]) {
                                setXsFunc(elem);
                            }
                            else {
                                visited[elem] = true;
                                stack.push(elem);
                                stack = stack.concat(nextNodesFunc(elem));
                            }
                            elem = stack.pop();
                        }
                    };
                    // First pass, assign smallest coordinates
                    const pass1 = (elem) => {
                        xs[elem] = blockG.inEdges(elem).reduce((acc, e) => Math.max(acc, xs[e.v] + blockG.edge(e)), 0);
                    };
                    // Second pass, assign greatest coordinates
                    const pass2 = (elem) => {
                        const min = blockG.outEdges(elem).reduce((acc, e) => Math.min(acc, xs[e.w] - blockG.edge(e)), Number.POSITIVE_INFINITY);
                        const node = g.node(elem);
                        if (min !== Number.POSITIVE_INFINITY && node.borderType !== borderType) {
                            xs[elem] = Math.max(xs[elem], min);
                        }
                    };
                    iterate(pass1, blockG.predecessors.bind(blockG));
                    iterate(pass2, blockG.successors.bind(blockG));
                    // Assign x coordinates to all nodes
                    for (const v of Object.values(align)) {
                        xs[v] = xs[root[v]];
                    }
                    return xs;
                };
                const buildBlockGraph = (g, layering, root, reverseSep) => {
                    const sep = (nodeSep, edgeSep, reverseSep) => {
                        return function(g, v, w) {
                            const vLabel = g.node(v);
                            const wLabel = g.node(w);
                            let sum = 0;
                            let delta;
                            sum += vLabel.width / 2;
                            if ('labelpos' in vLabel) {
                                switch (vLabel.labelpos.toLowerCase()) {
                                    case 'l': delta = -vLabel.width / 2; break;
                                    case 'r': delta = vLabel.width / 2; break;
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
                                switch (wLabel.labelpos.toLowerCase()) {
                                    case 'l': delta = wLabel.width / 2; break;
                                    case 'r': delta = -wLabel.width / 2; break;
                                }
                            }
                            if (delta) {
                                sum += reverseSep ? delta : -delta;
                            }
                            delta = 0;
                            return sum;
                        };
                    };
                    const blockGraph = new dagre.Graph();
                    const graphLabel = g.graph();
                    const sepFn = sep(graphLabel.nodesep, graphLabel.edgesep, reverseSep);
                    layering.forEach(function(layer) {
                        let u;
                        for (const v of layer) {
                            const vRoot = root[v];
                            blockGraph.setNode(vRoot);
                            if (u) {
                                const uRoot = root[u];
                                const prevMax = blockGraph.edge(uRoot, vRoot);
                                blockGraph.setEdge(uRoot, vRoot, Math.max(sepFn(g, v, u), prevMax || 0));
                            }
                            u = v;
                        }
                    });
                    return blockGraph;
                };

                // Returns the alignment that has the smallest width of the given alignments.
                const findSmallestWidthAlignment = (g, xss) => {
                    let minKey = Number.POSITIVE_INFINITY;
                    let minValue = undefined;
                    for (const xs of Object.values(xss)) {
                        let max = Number.NEGATIVE_INFINITY;
                        let min = Number.POSITIVE_INFINITY;
                        for (const entry of Object.entries(xs)) {
                            const v = entry[0];
                            const x = entry[1];
                            const halfWidth = g.node(v).width / 2;
                            max = Math.max(x + halfWidth, max);
                            min = Math.min(x - halfWidth, min);
                        }
                        const key = max - min;
                        if (key < minKey) {
                            minKey = key;
                            minValue = xs;
                        }
                    }
                    return minValue;
                };
                const balance = (xss, align) => {
                    const value = {};
                    if (align) {
                        const xs = xss[align.toLowerCase()];
                        for (const v of Object.keys(xss.ul)) {
                            value[v] = xs[v];
                        }
                    }
                    else {
                        for (const v of Object.keys(xss.ul)) {
                            const xs = [ xss.ul[v], xss.ur[v], xss.dl[v], xss.dr[v] ].sort((a, b) => a - b);
                            value[v] = (xs[1] + xs[2]) / 2;
                        }
                    }
                    return value;
                };

                /*
                * Marks all edges in the graph with a type-1 conflict with the 'type1Conflict'
                * property. A type-1 conflict is one where a non-inner segment crosses an
                * inner segment. An inner segment is an edge with both incident nodes marked
                * with the 'dummy' property.
                *
                * This algorithm scans layer by layer, starting with the second, for type-1
                * conflicts between the current layer and the previous layer. For each layer
                * it scans the nodes from left to right until it reaches one that is incident
                * on an inner segment. It then scans predecessors to determine if they have
                * edges that cross that inner segment. At the end a final scan is done for all
                * nodes on the current rank to see if they cross the last visited inner
                * segment.
                *
                * This algorithm (safely) assumes that a dummy node will only be incident on a
                * single node in the layers being scanned.
                */
                const findType1Conflicts = (g, layering) => {
                    const conflicts = {};
                    const visitLayer = (prevLayer, layer) => {
                        // last visited node in the previous layer that is incident on an inner
                        // segment.
                        let k0 = 0;
                        // Tracks the last node in this layer scanned for crossings with a type-1
                        // segment.
                        let scanPos = 0;
                        const prevLayerLength = prevLayer.length;
                        const lastNode = layer[layer.length - 1];
                        layer.forEach(function(v, i) {
                            const w = findOtherInnerSegmentNode(g, v);
                            const k1 = w ? g.node(w).order : prevLayerLength;
                            if (w || v === lastNode) {
                                for (const scanNode of layer.slice(scanPos, i + 1)) {
                                    for (const u of g.predecessors(scanNode)) {
                                        const uLabel = g.node(u);
                                        const uPos = uLabel.order;
                                        if ((uPos < k0 || k1 < uPos) &&
                                                !(uLabel.dummy && g.node(scanNode).dummy)) {
                                            addConflict(conflicts, u, scanNode);
                                        }
                                    }
                                }
                                scanPos = i + 1;
                                k0 = k1;
                            }
                        });
                        return layer;
                    };
                    if (layering.length > 0) {
                        layering.reduce(visitLayer);
                    }
                    return conflicts;
                };

                const findType2Conflicts = (g, layering) => {
                    const conflicts = {};
                    function scan(south, southPos, southEnd, prevNorthBorder, nextNorthBorder) {
                        let v;
                        for (let i = southPos; i < southEnd; i++) {
                            v = south[i];
                            if (g.node(v).dummy) {
                                for (const u of g.predecessors(v)) {
                                    const uNode = g.node(u);
                                    if (uNode.dummy && (uNode.order < prevNorthBorder || uNode.order > nextNorthBorder)) {
                                        addConflict(conflicts, u, v);
                                    }
                                }
                            }
                        }
                    }
                    function visitLayer(north, south) {
                        let prevNorthPos = -1;
                        let nextNorthPos;
                        let southPos = 0;
                        south.forEach(function(v, southLookahead) {
                            if (g.node(v).dummy === 'border') {
                                const predecessors = g.predecessors(v);
                                if (predecessors.length) {
                                    nextNorthPos = g.node(predecessors[0]).order;
                                    scan(south, southPos, southLookahead, prevNorthPos, nextNorthPos);
                                    southPos = southLookahead;
                                    prevNorthPos = nextNorthPos;
                                }
                            }
                            scan(south, southPos, south.length, nextNorthPos, north.length);
                        });
                        return south;
                    }
                    if (layering.length > 0) {
                        layering.reduce(visitLayer);
                    }
                    return conflicts;
                };

                const layering = buildLayerMatrix(g);
                const conflicts = Object.assign(findType1Conflicts(g, layering), findType2Conflicts(g, layering));
                const xss = {};
                for (const vert of ['u', 'd']) {
                    let adjustedLayering = vert === 'u' ? layering : Object.values(layering).reverse();
                    for (const horiz of ['l', 'r']) {
                        if (horiz === 'r') {
                            adjustedLayering = adjustedLayering.map((inner) => Object.values(inner).reverse());
                        }
                        const neighborFn = (vert === 'u' ? g.predecessors : g.successors).bind(g);
                        const align = verticalAlignment(g, adjustedLayering, conflicts, neighborFn);
                        const xs = horizontalCompaction(g, adjustedLayering, align.root, align.align, horiz === 'r');
                        if (horiz === 'r') {
                            for (const entry of Object.entries(xs)) {
                                xs[entry[0]] = -entry[1];
                            }
                        }
                        xss[vert + horiz] = xs;
                    }
                }

                const smallestWidth = findSmallestWidthAlignment(g, xss);
                /*
                * Align the coordinates of each of the layout alignments such that
                * left-biased alignments have their minimum coordinate at the same point as
                * the minimum coordinate of the smallest width alignment and right-biased
                * alignments have their maximum coordinate at the same point as the maximum
                * coordinate of the smallest width alignment.
                */
                const alignCoordinates = (xss, alignTo) => {
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
                        return [ min, max ];
                    };
                    const alignToRange = range(Object.values(alignTo));
                    for (const vert of ['u', 'd']) {
                        for (const horiz of ['l', 'r']) {
                            const alignment = vert + horiz;
                            const xs = xss[alignment];
                            let delta;
                            if (xs !== alignTo) {
                                const vsValsRange = range(Object.values(xs));
                                delta = horiz === 'l' ? alignToRange[0] - vsValsRange[0] : alignToRange[1] - vsValsRange[1];
                                if (delta) {
                                    const list = {};
                                    for (const key of Object.keys(xs)) {
                                        list[key] = xs[key] + delta;
                                    }
                                    xss[alignment] = list; //_.mapValues(xs, function(x) { return x + delta; });
                                }
                            }
                        }
                    }
                };
                alignCoordinates(xss, smallestWidth);
                return balance(xss, g.graph().align);
            };

            g = util_asNonCompoundGraph(g);
            const layering = buildLayerMatrix(g);
            const rankSep = g.graph().ranksep;
            let prevY = 0;
            for (const layer of layering) {
                const heights = layer.map((v) => g.node(v).height);
                const maxHeight = Math.max(...heights);
                for (const v of layer) {
                    g.node(v).y = prevY + maxHeight / 2;
                }
                prevY += maxHeight + rankSep;
            }
            for (const entry of Object.entries(positionX(g))) {
                g.node(entry[0]).x = entry[1];
            }
        };

        const positionSelfEdges = (g) => {
            for (const v of g.nodes()) {
                const node = g.node(v);
                if (node.dummy === 'selfedge') {
                    const selfNode = g.node(node.e.v);
                    const x = selfNode.x + selfNode.width / 2;
                    const y = selfNode.y;
                    const dx = node.x - x;
                    const dy = selfNode.height / 2;
                    g.setEdge(node.e, node.label);
                    g.removeNode(v);
                    node.label.points = [
                        { x: x + 2 * dx / 3, y: y - dy },
                        { x: x + 5 * dx / 6, y: y - dy },
                        { x: x +     dx    , y: y },
                        { x: x + 5 * dx / 6, y: y + dy },
                        { x: x + 2 * dx / 3, y: y + dy }
                    ];
                    node.label.x = node.x;
                    node.label.y = node.y;
                }
            }
        };

        const removeBorderNodes = (g) => {
            for (const v of g.nodes()) {
                if (g.children(v).length) {
                    const node = g.node(v);
                    const t = g.node(node.borderTop);
                    const b = g.node(node.borderBottom);
                    const l = g.node(node.borderLeft[node.borderLeft.length - 1]);
                    const r = g.node(node.borderRight[node.borderRight.length - 1]);
                    node.width = Math.abs(r.x - l.x);
                    node.height = Math.abs(b.y - t.y);
                    node.x = l.x + node.width / 2;
                    node.y = t.y + node.height / 2;
                }
            }
            for (const v of g.nodes()) {
                if (g.node(v).dummy === 'border') {
                    g.removeNode(v);
                }
            }
        };

        const fixupEdgeLabelCoords = (g) => {
            for (const e of g.edges()) {
                const edge = g.edge(e);
                if ('x' in edge) {
                    if (edge.labelpos === 'l' || edge.labelpos === 'r') {
                        edge.width -= edge.labeloffset;
                    }
                    switch (edge.labelpos) {
                        case 'l': edge.x -= edge.width / 2 + edge.labeloffset; break;
                        case 'r': edge.x += edge.width / 2 + edge.labeloffset; break;
                    }
                }
            }
        };

        const translateGraph = (g) => {
            let minX = Number.POSITIVE_INFINITY;
            let maxX = 0;
            let minY = Number.POSITIVE_INFINITY;
            let maxY = 0;
            const graphLabel = g.graph();
            const marginX = graphLabel.marginx || 0;
            const marginY = graphLabel.marginy || 0;
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
            for (const v of g.nodes()) {
                getExtremes(g.node(v));
            }
            for (const e of g.edges()) {
                const edge = g.edge(e);
                if ('x' in edge) {
                    getExtremes(edge);
                }
            }
            minX -= marginX;
            minY -= marginY;
            for (const v of g.nodes()) {
                const node = g.node(v);
                node.x -= minX;
                node.y -= minY;
            }
            for (const e of g.edges()) {
                const edge = g.edge(e);
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
            graphLabel.width = maxX - minX + marginX;
            graphLabel.height = maxY - minY + marginY;
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
                let w = rect.width / 2;
                let h = rect.height / 2;
                if (!dx && !dy) {
                    throw new Error('Not possible to find intersection inside of the rectangle');
                }
                let sx;
                let sy;
                if (Math.abs(dy) * w > Math.abs(dx) * h) {
                    // Intersection is top or bottom of rect.
                    if (dy < 0) {
                        h = -h;
                    }
                    sx = h * dx / dy;
                    sy = h;
                }
                else {
                    // Intersection is left or right of rect.
                    if (dx < 0) {
                        w = -w;
                    }
                    sx = w;
                    sy = w * dy / dx;
                }
                return { x: x + sx, y: y + sy };
            };
            for (const e of g.edges()) {
                const edge = g.edge(e);
                const nodeV = g.node(e.v);
                const nodeW = g.node(e.w);
                let p1;
                let p2;
                if (!edge.points) {
                    edge.points = [];
                    p1 = nodeW;
                    p2 = nodeV;
                }
                else {
                    p1 = edge.points[0];
                    p2 = edge.points[edge.points.length - 1];
                }
                edge.points.unshift(intersectRect(nodeV, p1));
                edge.points.push(intersectRect(nodeW, p2));
            }
        };

        const reversePointsForReversedEdges = (g) => {
            for (const e of g.edges()) {
                const edge = g.edge(e);
                if (edge.reversed) {
                    edge.points.reverse();
                }
            }
        };

        time('    makeSpaceForEdgeLabels',  function() { makeSpaceForEdgeLabels(g); });
        time('    removeSelfEdges',         function() { removeSelfEdges(g); });
        time('    acyclic_run',             function() { acyclic_run(g); });
        time('    nestingGraph_run',        function() { nestingGraph_run(g); });
        time('    rank',                    function() { rank(util_asNonCompoundGraph(g)); });
        time('    injectEdgeLabelProxies',  function() { injectEdgeLabelProxies(g); });
        time('    removeEmptyRanks',        function() { removeEmptyRanks(g); });
        time('    nestingGraph_cleanup',    function() { nestingGraph_cleanup(g); });
        time('    normalizeRanks',          function() { normalizeRanks(g); });
        time('    assignRankMinMax',        function() { assignRankMinMax(g); });
        time('    removeEdgeLabelProxies',  function() { removeEdgeLabelProxies(g); });
        time('    normalize',               function() { normalize(g); });
        time('    parentDummyChains',       function() { parentDummyChains(g); });
        time('    addBorderSegments',       function() { addBorderSegments(g); });
        time('    order',                   function() { order(g); });
        time('    insertSelfEdges',         function() { insertSelfEdges(g); });
        time('    coordinateSystem_adjust', function() { coordinateSystem_adjust(g); });
        time('    position',                function() { position(g); });
        time('    positionSelfEdges',       function() { positionSelfEdges(g); });
        time('    removeBorderNodes',       function() { removeBorderNodes(g); });
        time('    denormalize',             function() { denormalize(g); });
        time('    fixupEdgeLabelCoords',    function() { fixupEdgeLabelCoords(g); });
        time('    CoordinateSystem_undo',   function() { coordinateSystem_undo(g); });
        time('    translateGraph',          function() { translateGraph(g); });
        time('    assignNodeIntersects',    function() { assignNodeIntersects(g); });
        time('    reversePoints',           function() { reversePointsForReversedEdges(g); });
        time('    acyclic_undo',            function() { acyclic_undo(g); });
    };

    /*
    * Copies final layout information from the layout graph back to the input
    * graph. This process only copies whitelisted attributes from the layout graph
    * to the input graph, so it serves as a good place to determine what
    * attributes can influence layout.
    */
    const updateInputGraph = (inputGraph, layoutGraph) => {
        for (const v of inputGraph.nodes()) {
            const inputLabel = inputGraph.node(v);
            const layoutLabel = layoutGraph.node(v);
            if (inputLabel) {
                inputLabel.x = layoutLabel.x;
                inputLabel.y = layoutLabel.y;
                if (layoutGraph.children(v).length) {
                    inputLabel.width = layoutLabel.width;
                    inputLabel.height = layoutLabel.height;
                }
            }
        }
        for (const e of inputGraph.edges()) {
            const inputLabel = inputGraph.edge(e);
            const layoutLabel = layoutGraph.edge(e);
            inputLabel.points = layoutLabel.points;
            if ('x' in layoutLabel) {
                inputLabel.x = layoutLabel.x;
                inputLabel.y = layoutLabel.y;
            }
        }
        inputGraph.graph().width = layoutGraph.graph().width;
        inputGraph.graph().height = layoutGraph.graph().height;
    };

    time('layout', function() {
        const layoutGraph =
        time('  buildLayoutGraph', function() { return buildLayoutGraph(graph); });
        time('  runLayout',        function() { runLayout(layoutGraph, time); });
        time('  updateInputGraph', function() { updateInputGraph(graph, layoutGraph); });
    });
};

dagre.Graph = class {

    constructor(options) {
        options = options || {};
        this._isDirected = 'directed' in options ? options.directed : true;
        this._isMultigraph = 'multigraph' in options ? options.multigraph : false;
        this._isCompound = 'compound' in options ? options.compound : false;
        this._label = undefined;
        this._defaultNodeLabelFn = () => undefined;
        this._defaultEdgeLabelFn = () => undefined;
        this._nodes = {};
        if (this._isCompound) {
            this._parent = {};
            this._children = {};
            this._children[this.GRAPH_NODE] = {};
        }
        this._in = {};
        this._predecessors = {};
        this._out = {};
        this._successors = {};
        this._edgeObjs = {};
        this._edgeLabels = {};
        this._nodeCount = 0;
        this._edgeCount = 0;
    }

    isDirected() {
        return this._isDirected;
    }

    isMultigraph() {
        return this._isMultigraph;
    }

    isCompound() {
        return this._isCompound;
    }

    setGraph(label) {
        this._label = label;
    }

    graph() {
        return this._label;
    }

    setDefaultNodeLabel(newDefault) {
        this._defaultNodeLabelFn = newDefault;
        return this;
    }

    nodeCount() {
        return this._nodeCount;
    }

    nodes() {
        return Object.keys(this._nodes);
    }

    sources() {
        return this.nodes().filter((v) => {
            const value = this._in[v];
            return value && Object.keys(value).length === 0 && value.constructor === Object;
        });
    }

    setNode(v, value) {
        if (v in this._nodes) {
            if (arguments.length > 1) {
                this._nodes[v] = value;
            }
            return this;
        }
        this._nodes[v] = arguments.length > 1 ? value : this._defaultNodeLabelFn(v);
        if (this._isCompound) {
            this._parent[v] = this.GRAPH_NODE;
            this._children[v] = {};
            this._children[this.GRAPH_NODE][v] = true;
        }
        this._in[v] = {};
        this._predecessors[v] = {};
        this._out[v] = {};
        this._successors[v] = {};
        ++this._nodeCount;
        return this;
    }

    node(v) {
        return this._nodes[v];
    }

    hasNode(v) {
        return v in this._nodes;
    }

    removeNode(v) {
        if (v in this._nodes) {
            delete this._nodes[v];
            if (this._isCompound) {
                delete this._children[this._parent[v]][v];
                delete this._parent[v];
                for (const child of this.children(v)) {
                    this.setParent(child);
                }
                delete this._children[v];
            }
            for (const e of Object.keys(this._in[v])) {
                this.removeEdge(this._edgeObjs[e]);
            }
            delete this._in[v];
            delete this._predecessors[v];
            for (const e of Object.keys(this._out[v])) {
                this.removeEdge(this._edgeObjs[e]);
            }
            delete this._out[v];
            delete this._successors[v];
            --this._nodeCount;
        }
        return this;
    }

    setParent(v, parent) {
        if (!this._isCompound) {
            throw new Error('Cannot set parent in a non-compound graph');
        }
        if (parent === undefined) {
            parent = this.GRAPH_NODE;
        }
        else {
            // Coerce parent to string
            parent += '';
            for (let ancestor = parent; ancestor !== undefined; ancestor = this.parent(ancestor)) {
                if (ancestor === v) {
                    throw new Error('Setting ' + parent+ ' as parent of ' + v + ' would create a cycle.');
                }
            }
            this.setNode(parent);
        }
        this.setNode(v);
        delete this._children[this._parent[v]][v];
        this._parent[v] = parent;
        this._children[parent][v] = true;
        return this;
    }

    parent(v) {
        if (this._isCompound) {
            const parent = this._parent[v];
            if (parent !== this.GRAPH_NODE) {
                return parent;
            }
        }
    }

    children(v) {
        if (v === undefined) {
            v = this.GRAPH_NODE;
        }
        if (this._isCompound) {
            const children = this._children[v];
            if (children) {
                return Object.keys(children);
            }
        }
        else if (v === this.GRAPH_NODE) {
            return this.nodes();
        }
        else if (this.hasNode(v)) {
            return [];
        }
    }

    predecessors(v) {
        const value = this._predecessors[v];
        if (value) {
            return Object.keys(value);
        }
    }

    successors(v) {
        const value = this._successors[v];
        if (value) {
            return Object.keys(value);
        }
    }

    neighbors(v) {
        const value = this.predecessors(v);
        if (value) {
            return Array.from(new Set(value.concat(this.successors(v))));
        }
    }

    edges() {
        return Object.values(this._edgeObjs);
    }

    // setEdge(v, w, [value, [name]])
    // setEdge({ v, w, [name] }, [value])
    setEdge() {
        let v;
        let w;
        let name;
        let value;
        let valueSpecified = false;
        const arg0 = arguments[0];
        if (typeof arg0 === 'object' && arg0 !== null && 'v' in arg0) {
            v = arg0.v;
            w = arg0.w;
            name = arg0.name;
            if (arguments.length === 2) {
                value = arguments[1];
                valueSpecified = true;
            }
        }
        else {
            v = arg0;
            w = arguments[1];
            name = arguments[3];
            if (arguments.length > 2) {
                value = arguments[2];
                valueSpecified = true;
            }
        }
        v = '' + v;
        w = '' + w;
        if (name !== undefined) {
            name = '' + name;
        }
        const e = this.edgeArgsToId(this._isDirected, v, w, name);
        if (e in this._edgeLabels) {
            if (valueSpecified) {
                this._edgeLabels[e] = value;
            }
            return this;
        }
        if (name !== undefined && !this._isMultigraph) {
            throw new Error('Cannot set a named edge when isMultigraph = false');
        }
        // It didn't exist, so we need to create it.
        // First ensure the nodes exist.
        this.setNode(v);
        this.setNode(w);
        this._edgeLabels[e] = valueSpecified ? value : this._defaultEdgeLabelFn(v, w, name);
        v = '' + v;
        w = '' + w;
        if (!this._isDirected && v > w) {
            const tmp = v;
            v = w;
            w = tmp;
        }
        const edgeObj = name ? { v: v, w: w, name: name } : { v: v, w: w };
        Object.freeze(edgeObj);
        this._edgeObjs[e] = edgeObj;
        const incrementOrInitEntry = (map, k) => {
            if (map[k]) {
                map[k]++;
            }
            else {
                map[k] = 1;
            }
        };
        incrementOrInitEntry(this._predecessors[w], v);
        incrementOrInitEntry(this._successors[v], w);
        this._in[w][e] = edgeObj;
        this._out[v][e] = edgeObj;
        this._edgeCount++;
        return this;
    }

    edge(v, w, name) {
        const key = (arguments.length === 1 ? this.edgeObjToId(this._isDirected, arguments[0]) : this.edgeArgsToId(this._isDirected, v, w, name));
        return this._edgeLabels[key];
    }

    hasEdge(v, w, name) {
        const key = (arguments.length === 1 ? this.edgeObjToId(this._isDirected, arguments[0]) : this.edgeArgsToId(this._isDirected, v, w, name));
        return key in this._edgeLabels;
    }

    removeEdge(v, w, name) {
        const key = (arguments.length === 1 ? this.edgeObjToId(this._isDirected, arguments[0]) : this.edgeArgsToId(this._isDirected, v, w, name));
        const edge = this._edgeObjs[key];
        if (edge) {
            v = edge.v;
            w = edge.w;
            delete this._edgeLabels[key];
            delete this._edgeObjs[key];
            const decrementOrRemoveEntry = (map, k) => {
                if (!--map[k]) {
                    delete map[k];
                }
            };
            decrementOrRemoveEntry(this._predecessors[w], v);
            decrementOrRemoveEntry(this._successors[v], w);
            delete this._in[w][key];
            delete this._out[v][key];
            this._edgeCount--;
        }
        return this;
    }

    inEdges(v, u) {
        const inV = this._in[v];
        if (inV) {
            const edges = Object.values(inV);
            if (!u) {
                return edges;
            }
            return edges.filter((edge) => edge.v === u);
        }
    }

    outEdges(v, w) {
        const outV = this._out[v];
        if (outV) {
            const edges = Object.values(outV);
            if (!w) {
                return edges;
            }
            return edges.filter((edge) => edge.w === w);
        }
    }

    nodeEdges(v, w) {
        const inEdges = this.inEdges(v, w);
        if (inEdges) {
            return inEdges.concat(this.outEdges(v, w));
        }
    }

    edgeArgsToId(isDirected, v_, w_, name) {
        let v = '' + v_;
        let w = '' + w_;
        if (!isDirected && v > w) {
            const tmp = v;
            v = w;
            w = tmp;
        }
        return v + '\x01' + w + '\x01' + (name === undefined ? '\x00' : name);
    }

    edgeObjToId(isDirected, edgeObj) {
        return this.edgeArgsToId(isDirected, edgeObj.v, edgeObj.w, edgeObj.name);
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports = dagre;
}
