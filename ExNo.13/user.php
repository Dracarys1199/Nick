<style>
table, th, td {
  border: 1px solid black;
}
</style>
<form method="get">
  <label for="userId">User ID:</label>
  <input type="text" id="userId" name="userId">
  <button type="submit">Submit</button>
</form>

<!-- PHP code to display the XML data -->
<?php
// check if the user ID was submitted
if (isset($_GET['userId'])) {
  // get the user ID from the form
  $userId = $_GET['userId'];

  // load the XML file
  $xml = simplexml_load_file('user.xml');

  // use XPath to select the user data from the XML
  $userData = $xml->xpath("//user[userid='" . $userId . "']");

  // check if user data was found
  if (!empty($userData)) {
    // display the user data in an HTML table
    echo "<table>";
    echo "<tr><th>Username</th><th>Email</th><th>Address</th><th>Phone</th></tr>";
    foreach ($userData as $user) {
      echo "<tr><td>" . $user->username . "</td><td>" . $user->email . "</td><td>". $user->address .
	  "</td><td>" . $user->phone ."</td></tr>";
    }
    echo "</table>";
  } else {
    // display an error message if user data was not found
    echo "User data not found.";
  }
}
?>