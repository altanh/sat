#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <tuple>
#include <cassert>
#include <getopt.h>
#include <cstring>
#include <chrono>

using ident_t = int32_t;

struct lit_t
{
    ident_t var;
    bool polarity;

    ident_t sign() const
    {
        return polarity ? 1 : -1;
    }

    ident_t to_id() const
    {
        return var * sign();
    }
};

lit_t to_lit(ident_t l)
{
    return l > 0 ? lit_t{l, true} : lit_t{-l, false};
}

struct CNF
{
    ident_t v;
    ident_t c;
    uint64_t nnz;
    std::vector<std::vector<ident_t>> VC; // NOTE: 1-indexed
    std::vector<std::vector<ident_t>> CV; // NOTE: 1-indexed

    CNF() {}

    CNF(std::ifstream *in)
    {
        load(in);
    }

    void load(std::ifstream *in)
    {
        std::string line;
        nnz = 0;
        ident_t cur_clause = 1;
        while (std::getline(*in, line))
        {
            if (line[0] == 'p')
            {
                std::istringstream iss(line);
                std::string token;
                iss >> token; // p
                iss >> token;
                // check that it is a cnf file
                if (token != "cnf")
                {
                    std::cerr << "Error: not a cnf file" << std::endl;
                    return;
                }
                iss >> v;
                iss >> c;
                VC.resize(v + 1);
                CV.resize(c + 1);
            }
            else if (line[0] == 'c')
            {
                continue;
            }
            else
            {
                // read clause
                std::istringstream iss(line);
                ident_t token;
                while (iss >> token)
                {
                    if (token == 0)
                    {
                        break;
                    }
                    nnz++;

                    lit_t lit = to_lit(token);

                    VC[lit.var].push_back(cur_clause * lit.sign());
                    CV[cur_clause].push_back(lit.to_id());
                }
                cur_clause++;
            }
        }
    }
};

void write_mtx_header(std::ofstream *out, uint32_t n, uint32_t nnz, bool symmetric, bool integer)
{
    *out << "%%MatrixMarket matrix coordinate " << (integer ? "integer" : "real") << " " << (symmetric ? "symmetric" : "general") << std::endl;
    *out << n << " " << n << " " << nnz << std::endl;
}

class Timer
{
public:
    Timer(double *elapsed) : elapsed(elapsed)
    {
        start = std::chrono::high_resolution_clock::now();
    }

    ~Timer()
    {
        auto end = std::chrono::high_resolution_clock::now();
        *elapsed = std::chrono::duration<double>(end - start).count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    double *elapsed;
};

class FactorGraph
{
public:
    FactorGraph(const CNF &cnf, bool directed) : cnf(cnf), directed(directed) {}

    void write_mtx(std::ofstream *out)
    {
        bool symmetric = !directed;
        write_mtx_header(out, cnf.v + cnf.c, cnf.nnz, symmetric, true);

        double seconds;
        {
            Timer timer(&seconds);
            for (ident_t c = 1; c < cnf.c + 1; c++)
            {
                for (ident_t l : cnf.CV[c])
                {
                    ident_t c_idx = c + cnf.v;
                    lit_t lit = to_lit(l);
                    if (!directed)
                    {
                        // lower-triangular
                        *out << c_idx << " " << lit.var << " " << lit.sign() << std::endl;
                    }
                    else
                    {
                        // directed
                        if (lit.polarity)
                        {
                            *out << lit.var << " " << c_idx << " 1" << std::endl;
                        }
                        else
                        {
                            *out << c_idx << " " << lit.var << " 1" << std::endl;
                        }
                    }
                }
            }
        }
        std::cout << "Summary: wrote " << cnf.nnz << " edges and " << cnf.v + cnf.c << " vertices in " << seconds << " seconds" << std::endl;
    }

private:
    const CNF &cnf;
    const bool directed;
};

// Clause-incidence graph: vertices are clauses, edges connect clauses that share a variable.
// Additional information about adjacent clauses is stored in the edge weights.
class ClauseGraph
{
public:
    enum EdgeData
    {
        HEAD,
        TAIL,
        NUM_SHARED_VARIABLES,
        NUM_SHARED_LITERALS,
        SUBSUMPTION,
        NO_EDGE_DATA,
    };

    static constexpr const char *edge_data_names[NO_EDGE_DATA] = {
        "head",
        "tail",
        "num_shared_var",
        "num_shared_lit",
        "subsumption",
    };

    enum VertexData
    {
        NUM_LITERALS = 0,
        AVERAGE_POLARITY,
        NO_VERTEX_DATA,
    };

    static constexpr const char *vertex_data_names[NO_VERTEX_DATA] = {
        "num_lit",
        "avg_polarity",
    };

    ClauseGraph(const CNF &cnf) : cnf(cnf)
    {
        build();
    }

    void write_mtx(std::ofstream *out, EdgeData data_kind)
    {
        double seconds;
        {
            Timer timer(&seconds);
            write_mtx_header(out, cnf.c, nnz, true, data_kind == NO_EDGE_DATA);
            for (uint64_t eid = 0; eid < nnz; ++eid)
            {
                *out << edge_data[HEAD][eid] << " " << edge_data[TAIL][eid] << " ";
                if (data_kind != NO_EDGE_DATA)
                {
                    *out << edge_data[data_kind][eid];
                }
                else
                {
                    *out << "1";
                }
                *out << std::endl;
            }
        }
        std::cout << "Summary: wrote " << nnz << " edges and " << cnf.c << " vertices in " << seconds << " seconds" << std::endl;
    }

    void write_edge_data(std::ofstream *out)
    {
        double seconds;
        {
            Timer timer(&seconds);
            // write csv header
            for (int i = 0; i < NO_EDGE_DATA; ++i)
            {
                *out << edge_data_names[i];
                if (i < NO_EDGE_DATA - 1)
                {
                    *out << ",";
                }
            }
            *out << std::endl;

            for (uint64_t edge_idx = 0; edge_idx < nnz; edge_idx++)
            {
                for (int i = 0; i < NO_EDGE_DATA; ++i)
                {
                    *out << edge_data[i][edge_idx];
                    if (i < NO_EDGE_DATA - 1)
                    {
                        *out << ",";
                    }
                }
                *out << std::endl;
            }
        }
        std::cout << "Summary: wrote " << nnz << " rows of edge data in " << seconds << " seconds" << std::endl;
    }

    void write_vertex_data(std::ofstream *out)
    {
        double seconds;
        {
            Timer timer(&seconds);
            // write csv header
            *out << "clause_id,";
            for (int i = 0; i < NO_VERTEX_DATA; ++i)
            {
                *out << vertex_data_names[i];
                if (i < NO_VERTEX_DATA - 1)
                {
                    *out << ",";
                }
            }
            *out << std::endl;

            for (ident_t c = 1; c <= cnf.c; c++)
            {
                *out << c << ",";
                for (int i = 0; i < NO_VERTEX_DATA; ++i)
                {
                    *out << vertex_data[i][c];
                    if (i < NO_VERTEX_DATA - 1)
                    {
                        *out << ",";
                    }
                }
                *out << std::endl;
            }
        }
        std::cout << "Summary: wrote " << cnf.c << " rows of vertex data in " << seconds << " seconds" << std::endl;
    }

private:
    const CNF &cnf;
    uint64_t nnz;
    std::vector<std::set<ident_t>> CC; // NOTE: ordered set so iterating over edges is deterministic
    std::vector<double> edge_data[NO_EDGE_DATA];
    std::vector<double> vertex_data[NO_VERTEX_DATA];

    void build()
    {
        CC.resize(cnf.c + 1);
        nnz = 0;

        double seconds;
        {
            Timer timer(&seconds);
            for (ident_t v = 1; v <= cnf.v; v++)
            {
                auto &clauses = cnf.VC[v];

                for (ident_t j = 0; j < clauses.size(); j++)
                {
                    for (ident_t k = j + 1; k < clauses.size(); k++)
                    {
                        ident_t c1 = to_lit(clauses[j]).var;
                        ident_t c2 = to_lit(clauses[k]).var;

                        // only store lower triangle
                        if (c1 < c2)
                        {
                            std::swap(c1, c2);
                        }

                        bool inserted = CC[c1].insert(c2).second;
                        if (inserted)
                        {
                            nnz++;
                        }
                    }
                }
            }
        }
        std::cout << "Clause graph construction time: " << seconds << "s" << std::endl;

        uint64_t edge_idx = 0;

        {
            Timer timer(&seconds);
            // compute edge data
            for (int i = 0; i < NO_EDGE_DATA; ++i)
            {
                edge_data[i].resize(nnz);
            }

            for (ident_t c1 = 1; c1 <= cnf.c; c1++)
            {
                for (ident_t c2 : CC[c1])
                {
                    uint32_t num_shared_vars = 0;
                    uint32_t num_shared_lits = 0;
                    std::tie(num_shared_vars, num_shared_lits) = num_shared(c1, c2);

                    // subsumption is defined as num_shared_lits / min(num_lits(c1), num_lits(c2))
                    double subsumption = (double)num_shared_lits / std::min(cnf.CV[c1].size(), cnf.CV[c2].size());

                    edge_data[HEAD][edge_idx] = c1;
                    edge_data[TAIL][edge_idx] = c2;
                    edge_data[NUM_SHARED_VARIABLES][edge_idx] = num_shared_vars;
                    edge_data[NUM_SHARED_LITERALS][edge_idx] = num_shared_lits;
                    edge_data[SUBSUMPTION][edge_idx] = subsumption;

                    edge_idx++;
                }
            }
        }
        std::cout << "Edge data computation time: " << seconds << "s" << std::endl;

        assert(edge_idx == nnz);

        {
            Timer timer(&seconds);
            // compute vertex data
            for (int i = 0; i < NO_VERTEX_DATA; ++i)
            {
                vertex_data[i].resize(cnf.c + 1);
            }

            for (ident_t c = 1; c <= cnf.c; c++)
            {
                auto &lits = cnf.CV[c];
                double sum_polarity = 0;
                for (ident_t l : lits)
                {
                    sum_polarity += to_lit(l).sign();
                }

                vertex_data[NUM_LITERALS][c] = lits.size();
                vertex_data[AVERAGE_POLARITY][c] = sum_polarity / lits.size();
            }
        }
        std::cout << "Vertex data computation time: " << seconds << "s" << std::endl;
    }

    std::pair<uint32_t, uint32_t> num_shared(ident_t c1, ident_t c2)
    {
        auto &lits1 = cnf.CV[c1];
        auto &lits2 = cnf.CV[c2];

        uint32_t num_shared_vars = 0;
        uint32_t num_shared_lits = 0;
        for (ident_t l1 : lits1)
        {
            for (ident_t l2 : lits2)
            {
                if (l1 == l2)
                {
                    num_shared_lits++;
                    num_shared_vars++;
                }
                else if (to_lit(l1).var == to_lit(l2).var)
                {
                    num_shared_vars++;
                }
            }
        }
        return std::make_pair(num_shared_vars, num_shared_lits);
    }
};

// Variable-incidence graph: vertices are variables, edges connect variables that appear in the same clause.
class VariableGraph
{
public:
    enum EdgeData
    {
        HEAD,
        TAIL,
        NUM_SHARED_CLAUSES,
        NUM_SHARED_POLARITY,
        NO_EDGE_DATA,
    };

    static constexpr const char *edge_data_names[NO_EDGE_DATA] = {
        "head",
        "tail",
        "num_shared_clause",
        "num_shared_polarity",
    };

    enum VertexData
    {
        NUM_CLAUSES,
        AVERAGE_POLARITY,
        NO_VERTEX_DATA,
    };

    static constexpr const char *vertex_data_names[NO_VERTEX_DATA] = {
        "num_clause",
        "average_polarity",
    };

    VariableGraph(const CNF &cnf) : cnf(cnf)
    {
        build();
    }

    void write_mtx(std::ofstream *out, EdgeData data_kind)
    {
        double seconds;
        {
            write_mtx_header(out, cnf.v, nnz, true, data_kind == NO_EDGE_DATA);

            for (uint64_t eid = 0; eid < nnz; ++eid)
            {
                ident_t v1 = static_cast<ident_t>(edge_data[HEAD][eid]);
                ident_t v2 = static_cast<ident_t>(edge_data[TAIL][eid]);
                *out << v1 << " " << v2 << " ";
                if (data_kind != NO_EDGE_DATA)
                {
                    *out << edge_data[data_kind][eid];
                }
                else
                {
                    *out << 1;
                }
                *out << std::endl;
            }
        }
        std::cout << "Summary: wrote " << nnz << " edges, " << cnf.v << " vertices in " << seconds << "seconds" << std::endl;
    }

    // TODO: deduplicate code

    void write_edge_data(std::ofstream *out)
    {
        double seconds;
        {
            Timer timer(&seconds);
            // write csv header
            for (int i = 0; i < NO_EDGE_DATA; ++i)
            {
                *out << edge_data_names[i];
                if (i < NO_EDGE_DATA - 1)
                {
                    *out << ",";
                }
            }
            *out << std::endl;

            for (uint64_t edge_idx = 0; edge_idx < nnz; edge_idx++)
            {
                for (int i = 0; i < NO_EDGE_DATA; ++i)
                {
                    *out << edge_data[i][edge_idx];
                    if (i < NO_EDGE_DATA - 1)
                    {
                        *out << ",";
                    }
                }
                *out << std::endl;
            }
        }

        std::cout << "Summary: wrote " << nnz << " rows of edge data in " << seconds << "seconds" << std::endl;
    }

    void write_vertex_data(std::ofstream *out)
    {
        double seconds;
        {
            Timer timer(&seconds);
            // write csv header
            *out << "clause_id,";
            for (int i = 0; i < NO_VERTEX_DATA; ++i)
            {
                *out << vertex_data_names[i];
                if (i < NO_VERTEX_DATA - 1)
                {
                    *out << ",";
                }
            }
            *out << std::endl;

            for (ident_t c = 1; c <= cnf.c; c++)
            {
                *out << c << ",";
                for (int i = 0; i < NO_VERTEX_DATA; ++i)
                {
                    *out << vertex_data[i][c];
                    if (i < NO_VERTEX_DATA - 1)
                    {
                        *out << ",";
                    }
                }
                *out << std::endl;
            }
        }
        std::cout << "Summary: wrote " << cnf.c << " rows of vertex data in " << seconds << "seconds" << std::endl;
    }

private:
    const CNF &cnf;
    uint64_t nnz;
    std::vector<std::set<ident_t>> VV;
    std::vector<double> edge_data[NO_EDGE_DATA];
    std::vector<double> vertex_data[NO_VERTEX_DATA];

    void build()
    {
        VV.resize(cnf.v + 1);
        nnz = 0;

        double seconds;
        {
            Timer timer(&seconds);
            for (ident_t c = 1; c <= cnf.c; c++)
            {
                auto &lits = cnf.CV[c];

                for (ident_t l1 = 0; l1 < lits.size(); l1++)
                {
                    for (ident_t l2 = l1 + 1; l2 < lits.size(); l2++)
                    {
                        ident_t v1 = to_lit(lits[l1]).var;
                        ident_t v2 = to_lit(lits[l2]).var;

                        // only store lower triangle
                        if (v1 < v2)
                        {
                            std::swap(v1, v2);
                        }
                        bool inserted = VV[v1].insert(v2).second;
                        if (inserted)
                        {
                            nnz++;
                        }
                    }
                }
            }
        }
        std::cout << "Variable graph construction time: " << seconds << "s" << std::endl;

        // compute edge data
        for (int i = 0; i < NO_EDGE_DATA; ++i)
        {
            edge_data[i].resize(nnz);
        }

        uint64_t edge_idx = 0;

        {
            Timer timer(&seconds);
            for (ident_t v1 = 1; v1 <= cnf.v; v1++)
            {
                for (ident_t v2 : VV[v1])
                {
                    uint32_t num_shared_clauses = 0;
                    uint32_t num_shared_polarity = 0;
                    std::tie(num_shared_clauses, num_shared_polarity) = num_shared(v1, v2);

                    edge_data[HEAD][edge_idx] = v1;
                    edge_data[TAIL][edge_idx] = v2;
                    edge_data[NUM_SHARED_CLAUSES][edge_idx] = num_shared_clauses;
                    edge_data[NUM_SHARED_POLARITY][edge_idx] = num_shared_polarity;

                    edge_idx++;
                }
            }
        }
        std::cout << "Edge data computation time: " << seconds << "s" << std::endl;

        assert(edge_idx == nnz);

        // compute vertex data
        for (int i = 0; i < NO_VERTEX_DATA; ++i)
        {
            vertex_data[i].resize(cnf.v + 1);
        }

        {
            Timer timer(&seconds);
            for (ident_t v = 1; v <= cnf.v; v++)
            {
                auto &clauses = cnf.VC[v];
                double sum_polarity = 0;
                for (ident_t c : clauses)
                {
                    sum_polarity += to_lit(c).sign();
                }

                vertex_data[NUM_CLAUSES][v] = clauses.size();
                vertex_data[AVERAGE_POLARITY][v] = sum_polarity / clauses.size();
            }
        }
        std::cout << "Vertex data computation time: " << seconds << "s" << std::endl;
    }

    std::pair<uint32_t, uint32_t> num_shared(ident_t v1, ident_t v2)
    {
        auto &clauses1 = cnf.VC[v1];
        auto &clauses2 = cnf.VC[v2];

        uint32_t num_shared_clauses = 0;
        uint32_t num_shared_polarity = 0;
        for (ident_t c1 : clauses1)
        {
            for (ident_t c2 : clauses2)
            {
                if (c1 == c2)
                {
                    num_shared_polarity++;
                    num_shared_clauses++;
                }
                else if (to_lit(c1).var == to_lit(c2).var)
                {
                    num_shared_clauses++;
                }
            }
        }

        return std::make_pair(num_shared_clauses, num_shared_polarity);
    }
};

int lookup_string(const char *str, const char *const *strings, uint32_t no_strings)
{
    for (uint32_t i = 0; i < no_strings; ++i)
    {
        if (strcmp(str, strings[i]) == 0)
        {
            return i;
        }
    }

    return -1;
}

// TODO: parallelize using OpenMP

int main(int argc, char **argv)
{
    int c;
    char *cnf_file = NULL;
    char *graph_type = NULL;
    char *edge_data = NULL;
    char *save_edge = NULL;
    char *save_vertex = NULL;
    char *out_file = NULL;

    static const char *graph_types[] = {"factor", "variable", "clause"};
    int graph_type_id;

    while (true)
    {
        static struct option long_options[] =
            {
                {"help", no_argument, 0, 'h'},
                {"file", required_argument, 0, 'f'},
                {"graph-type", required_argument, 0, 't'},
                {"edge-data", required_argument, 0, 'e'},
                {"save-edge", required_argument, 0, 'E'},
                {"save-vertex", required_argument, 0, 'V'},
                {"out", required_argument, 0, 'o'},
                {0, 0, 0, 0}};

        int option_index = 0;
        c = getopt_long(argc, argv, "hf:t:e:o:", long_options, &option_index);

        if (c == -1)
            break;

        switch (c)
        {
        case 'h':
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -h, --help\t\t\tShow this help message and exit" << std::endl;
            std::cout << "  -f, --file\t\t\tCNF file to read, required" << std::endl;
            std::cout << "  -o, --out\t\t\tMatrixMarket file to write, required" << std::endl;
            std::cout << "  -t, --graph-type\t\tGraph type to output (factor, variable, clause), required" << std::endl;
            std::cout << "  -e, --edge-data\t\tEdge data to output" << std::endl;
            std::cout << "  --save-edge\t\t\tSave edge features to specified file" << std::endl;
            std::cout << "  --save-vertex\t\t\tSave vertex features to specified file" << std::endl;
            exit(0);
        case 'f':
            cnf_file = optarg;
            break;
        case 't':
            graph_type = optarg;
            break;
        case 'e':
            edge_data = optarg;
            break;
        case 'E':
            save_edge = optarg;
            break;
        case 'V':
            save_vertex = optarg;
            break;
        case 'o':
            out_file = optarg;
            break;
        default:
            exit(1);
        }
    }

    if (cnf_file == NULL)
    {
        std::cout << "CNF file not specified" << std::endl;
        exit(1);
    }

    if (out_file == NULL)
    {
        std::cout << "Output file not specified" << std::endl;
        exit(1);
    }

    if (graph_type == NULL)
    {
        std::cout << "Graph type not specified" << std::endl;
        exit(1);
    }
    else
    {
        graph_type_id = lookup_string(graph_type, graph_types, sizeof(graph_types));
        if (graph_type_id == -1)
        {
            std::cout << "Unknown graph type: " << graph_type << std::endl;
            exit(1);
        }
    }

    std::ifstream cnf_stream(cnf_file);
    if (!cnf_stream.is_open())
    {
        std::cout << "Could not open CNF file: " << cnf_file << std::endl;
        exit(1);
    }
    CNF cnf(&cnf_stream);

    std::ofstream out_stream(out_file);
    if (!out_stream.is_open())
    {
        std::cout << "Could not open output file: " << out_file << std::endl;
        exit(1);
    }

    std::ofstream edge_stream;
    if (save_edge != NULL)
    {
        edge_stream.open(save_edge);
        if (!edge_stream.is_open())
        {
            std::cout << "Could not open edge feature file: " << save_edge << std::endl;
            exit(1);
        }
    }

    std::ofstream vertex_stream;
    if (save_vertex != NULL)
    {
        vertex_stream.open(save_vertex);
        if (!vertex_stream.is_open())
        {
            std::cout << "Could not open vertex feature file: " << save_vertex << std::endl;
            exit(1);
        }
    }

    if (graph_type_id == 0)
    {
        FactorGraph G(cnf, false);
        G.write_mtx(&out_stream);
    }
    else if (graph_type_id == 1)
    {
        // check edge type
        int edge_type_id = VariableGraph::EdgeData::NO_EDGE_DATA;
        if (edge_data != NULL)
        {
            edge_type_id = lookup_string(edge_data, VariableGraph::edge_data_names, VariableGraph::EdgeData::NO_EDGE_DATA);
            if (edge_type_id == -1)
            {
                std::cout << "Unknown edge data type: " << edge_data << std::endl;
                exit(1);
            }
        }
        VariableGraph G(cnf);
        G.write_mtx(&out_stream, static_cast<VariableGraph::EdgeData>(edge_type_id));
        if (save_edge != NULL)
        {
            G.write_edge_data(&edge_stream);
        }
        if (save_vertex != NULL)
        {
            G.write_vertex_data(&vertex_stream);
        }
    }
    else if (graph_type_id == 2)
    {
        // check edge type
        int edge_type_id = ClauseGraph::EdgeData::NO_EDGE_DATA;
        if (edge_data != NULL)
        {
            edge_type_id = lookup_string(edge_data, ClauseGraph::edge_data_names, ClauseGraph::EdgeData::NO_EDGE_DATA);
            if (edge_type_id == -1)
            {
                std::cout << "Unknown edge data type: " << edge_data << std::endl;
                exit(1);
            }
        }
        ClauseGraph G(cnf);
        G.write_mtx(&out_stream, static_cast<ClauseGraph::EdgeData>(edge_type_id));
        if (save_edge != NULL)
        {
            G.write_edge_data(&edge_stream);
        }
        if (save_vertex != NULL)
        {
            G.write_vertex_data(&vertex_stream);
        }
    }

    return 0;
}
