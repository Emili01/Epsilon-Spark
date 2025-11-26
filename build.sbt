name := "epsilonClassifier"
version := "0.1.0"

scalaVersion := "2.12.18"

javacOptions ++= Seq("-source", "17", "-target", "17")
Compile / scalacOptions ++= Seq("-release", "17")

run / fork := true
run / javaOptions ++= Seq(
  "--add-exports=java.base/sun.nio.ch=ALL-UNNAMED",
  "--add-exports=java.base/java.nio=ALL-UNNAMED"
)

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core"   % "3.5.3",
  "org.apache.spark" %% "spark-sql"    % "3.5.3",
  "org.apache.spark" %% "spark-mllib"  % "3.5.3",
  "com.github.scopt" %% "scopt"        % "4.1.0"
)


Compile / mainClass := Some("com.epsilonClassifier.Main")


import sbtassembly.MergeStrategy
ThisBuild / assembly / assemblyMergeStrategy := {
  case PathList("META-INF", xs @ _*) =>
    xs match {
      case "services" :: _ => MergeStrategy.filterDistinctLines
      case _               => MergeStrategy.discard
    }
  case "reference.conf"    => MergeStrategy.concat
  case "application.conf"  => MergeStrategy.concat
  case x if x.endsWith(".proto")       => MergeStrategy.first
  case x if x.contains("module-info")  => MergeStrategy.discard
  case _                  => MergeStrategy.first
}
assembly / assemblyJarName := s"${name.value}-${version.value}.jar"
