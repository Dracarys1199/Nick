<%@ page import="java.sql.*" %>

<html>
<head>
    <title>Movie Details</title>
</head>
<body>
    <h1>Movie Details</h1>

    <table border="1">
        <tr>
            <th>Title</th>
            <th>Year</th>
            <th>Rating</th>
            <th>Country of Origin</th>
            <th>Comments</th>
            <th>Update</th>
        </tr>

        <% 
            try {
                // Establish a database connection
                Class.forName("com.mysql.jdbc.Driver");
                String url = "jdbc:mysql://localhost:3306/aai";
                String username = "root";
                String password = "";
                Connection conn = DriverManager.getConnection(url, username, password);

                // Create and execute the SQL query to fetch movie details
                String query = "SELECT title, year, rating, country_of_origin, comments FROM movies";
                Statement stmt = conn.createStatement();
                ResultSet rs = stmt.executeQuery(query);

                // Loop through the result set and display the movie details
                while (rs.next()) {
                    String title = rs.getString("title");
                    int year = rs.getInt("year");
                    String rating = rs.getString("rating");
                    String countryOfOrigin = rs.getString("country_of_origin");
                    String comments = rs.getString("comments");

                    %>
                    <tr>
                        <td><%= title %></td>
                        <td><%= year %></td>
                        <td><%= rating %></td>
                        <td><%= countryOfOrigin %></td>
                        <td>
                            <form action="updateComments.jsp" method="post">
                                <input type="hidden" name="title" value="<%= title %>">
                                <textarea name="comments"><%= comments %></textarea>
                                <button type="submit">Update</button>
                            </form>
                        </td>
                    </tr>
                    <%
                }

                // Close the database resources
                rs.close();
                stmt.close();
                conn.close();
            } catch (Exception e) {
                out.println("An error occurred: " + e.getMessage());
            }
        %>
    </table>

</body>
</html>