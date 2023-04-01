#include <iostream>
#include <fstream>
#include <sstream>

// Reads the header of the CNF file, which contains the number of variables and clauses, and
// counts the number of edges in the graph.
bool get_metadata(std::ifstream *in, int *v, int *c, int64_t *nnz)
{
    std::string line;
    *nnz = 0;
    while (std::getline(*in, line))
    {
        if (line[0] == 'p')
        {
            std::istringstream iss(line);
            std::string token;
            iss >> token;
            iss >> token;
            // check that it is a cnf file
            if (token != "cnf")
            {
                std::cerr << "Error: not a cnf file" << std::endl;
                return false;
            }
            iss >> *v;
            iss >> *c;
        }
        else if (line[0] == 'c')
        {
            continue;
        }
        else
        {
            std::istringstream iss(line);
            int token;
            while (iss >> token)
            {
                if (token == 0)
                {
                    break;
                }
                (*nnz)++;
            }
        }
    }
    return true;
}

// Given a CNF file with n variables and m clauses, we build the (bipartite) factor graph on
// n + m vertices, where an edge connects a variable to a clause if the variable appears in
// the clause. We offer two encodings of polarity:
//   1. By edge direction: if the variable appears positively in the clause, we add an edge
//      from the variable to the clause, and if it appears negatively, we add an edge from
//      the clause to the variable.
//   2. By edge weight: if the variable appears positively in the clause, we add an edge
//      from the variable to the clause with weight 1, and if it appears negatively, we add
//      an edge from the variable to the clause with weight -1.
// The first encoding produces a directed graph, while the second produces an undirected graph.
//
// We also offer the option to output all edges, in the case of the undirected output.
void write_factor_graph(std::ifstream *in, std::ofstream *out, int v, int c, bool polarity_by_edge, bool output_all_edges)
{
    std::string line;
    int clause = v;
    while (std::getline(*in, line))
    {
        if (line[0] == 'c' || line[0] == 'p')
        {
            continue;
        }
        std::istringstream iss(line);
        int token;
        while (iss >> token)
        {
            if (token == 0)
            {
                clause++;
                break;
            }
            // get polarity and abs value of token
            bool polarity = token > 0;
            token = token > 0 ? token : -token;
            if (polarity_by_edge)
            {
                if (polarity)
                {
                    *out << token << " " << clause << " 1" << std::endl;
                }
                else
                {
                    *out << clause << " " << token << " 1" << std::endl;
                }
            }
            else
            {
                // Lower triangle of the matrix (clause is always numbered higher than the variable).
                *out << clause << " " << token << " " << (polarity ? 1 : -1) << std::endl;
                // If we are outputting all edges, we also output the edge from the variable to the clause.
                if (output_all_edges)
                {
                    *out << token << " " << clause << " " << (polarity ? 1 : -1) << std::endl;
                }
            }
        }
    }
}

void write_mtx_header(std::ofstream *out, int v, int c, int64_t nnz, bool symmetric)
{
    *out << "%%MatrixMarket matrix coordinate integer " << (symmetric ? "symmetric" : "general") << std::endl;
    *out << v + c << " " << v + c << " " << nnz << std::endl;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <cnf file> [polarity by edge direction] [output all edges]" << std::endl;
        return 1;
    }
    std::ifstream in(argv[1]);
    if (!in.is_open())
    {
        std::cerr << "Error: could not open file " << argv[1] << std::endl;
        return 1;
    }
    std::ofstream out(argv[1] + std::string(".mtx"));
    if (!out.is_open())
    {
        std::cerr << "Error: could not open file " << argv[1] + std::string(".mtx") << std::endl;
        return 1;
    }
    bool polarity_by_edge = argc > 2 ? std::string(argv[2]) == "true" : false;
    bool output_all_edges = argc > 3 ? std::string(argv[3]) == "true" : false;
    int v, c;
    int64_t nnz;
    if (!get_metadata(&in, &v, &c, &nnz))
    {
        std::cerr << "Error: could not read header" << std::endl;
        return 1;
    }
    // print CNF metadata
    std::cout << "Number of variables: " << v << std::endl;
    std::cout << "Number of clauses: " << c << std::endl;
    std::cout << "Number of edges: " << nnz << std::endl;

    // reset input stream
    in.clear();
    in.seekg(0, std::ios::beg);

    write_mtx_header(&out, v, c, nnz, !output_all_edges);
    write_factor_graph(&in, &out, v, c, polarity_by_edge, output_all_edges);
    return 0;
}
