#include <R.h>
#include <Rinternals.h>
#include <vector>
#include <map>

using namespace std;

typedef pair<int,int> Connection;

// Edit By Mahdi
struct processedData{
  double* y;
  double* x;
  int* freq;
  int len;
};
// End of Edit by Mahdi
struct groupDataNode {
    bool active;
    double mu, lambda, deriv, mergeLambda, xmin, xmax;
    int grpSize, mergeTo;
    vector<int> neighbour;
};

// make a class for all the groups

class NIRClass {
    // used to describe nodes that are connected

    vector<groupDataNode> groupVec;
    multimap<double, Connection> groupMove; // multimap to store the times when groups hit each other
    int maxgroup;
    int numVariables;
    
    void checkInput(SEXP y); // checks if y has the right input format
    //Edit by Mahdi
    processedData processInput(SEXP y, SEXP x); // checks if y has the right input format
    //End of Edit by Mahdi
    void addConnection(int grpOne, int grpTwo, double lambda);
    double getCurrMu(groupDataNode x, double lambda) {return(x.mu + (lambda-x.lambda)*x.deriv);};
    vector<int> getNeighbours(int grpNum, int exclGrp); // get the neighbours of grpNum, excluding exclGrp
    void updateNeighbours(vector<int> updateGrp, int oldGrp, int newGrp); // changes the grpNumber of the neighbours from old to new
    void deactivateGroup(int grpNum, int newGrp, double lambda); // deactivate group grpNum, which merges into newGrp at lambda
    SEXP prepSolTree(int numGrps); // prepare the list for the solution tree
    
public:
    NIRClass(SEXP y); // Original Constructor
    NIRClass(SEXP y, SEXP x); // Mahdi's version of the Constructor
    void mergeGroups(int grpOne, int grpTwo, double lambda); // merges two groups
    pair<double, Connection> getNextConnection(); // returns the connection with the smallest lambda value
    SEXP solutionTree(); // returns the found solution so far in the form of a tree described in vector format
//    void printGroupVec();
//    void printGroupMove();
};


extern "C"
{

//main functions for NIR
SEXP NIR(SEXP y); // Originial NIR function
SEXP NIR2(SEXP y, SEXP x); // Mahdi's version which merge the instances that have duplicated x

// function that for a vector of lambdas explicitly calculates and returns the solution
SEXP NIRexplicitSolution(SEXP solTree, SEXP lambdaVec); 
}
