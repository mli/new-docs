stage("Document") {
  node {
    ws('workspace/new-docs') {
      checkout scm
      sh "build/build.sh"
    }
  }
}
