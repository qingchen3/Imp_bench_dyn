#include <iostream>

#include "include/kuzu.hpp"

using namespace kuzu::main;
using namespace std;

int main() {
    // Create an empty database.
    SystemConfig systemConfig;
    auto database = make_unique<Database>("test", systemConfig);

    // Connect to the database.
    auto connection = make_unique<Connection>(database.get());

    // Create the schema.
    connection->query("CREATE NODE TABLE User(name STRING, age INT64, PRIMARY KEY (name))");
    connection->query("CREATE NODE TABLE City(name STRING, population INT64, PRIMARY KEY (name))");
    connection->query("CREATE REL TABLE Follows(FROM User TO User, since INT64)");
    connection->query("CREATE REL TABLE LivesIn(FROM User TO City)");

    // Load data.
    connection->query("COPY User FROM \"user.csv\"");
    connection->query("COPY City FROM \"city.csv\"");
    connection->query("COPY Follows FROM \"follows.csv\"");
    connection->query("COPY LivesIn FROM \"lives-in.csv\"");

    // Execute a simple query.
    auto result =
        connection->query("MATCH (a:User)-[f:Follows]->(b:User) RETURN a.name, f.since, b.name;");

    // Output query result.
    while (result->hasNext()) {
        auto row = result->getNext();
        std::cout << row->getValue(0)->getValue<string>() << " "
                  << row->getValue(1)->getValue<int64_t>() << " "
                  << row->getValue(2)->getValue<string>() << std::endl;
    }
    return 0;
}
