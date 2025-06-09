"""
Test to verify that all critical imports work correctly in Docker environment.

This test prevents regressions where Docker containers fail to import modules
due to incorrect Python interpreter usage or missing dependencies.
"""

import subprocess
import pytest


def test_local_imports():
    """Test that critical imports work in local environment."""
    # Test core imports that were failing in production
    import mcp_synaptic
    from mcp_synaptic.rag.embeddings import EmbeddingManager
    from mcp_synaptic.rag.database.core import RAGDatabase
    from mcp_synaptic.core.server import SynapticServer
    from mcp_synaptic.config.settings import Settings
    
    # If we get here, all imports succeeded
    assert True


def test_docker_imports():
    """Test that critical imports work in Docker environment using correct Python interpreter."""
    # Build the Docker image first
    build_result = subprocess.run(
        ["docker-compose", "-f", "docker/docker-compose.yaml", "build", "mcp-synaptic"],
        capture_output=True,
        text=True,
        cwd="."
    )
    
    if build_result.returncode != 0:
        pytest.skip(f"Docker build failed: {build_result.stderr}")
    
    # Test that system python fails (this validates our troubleshooting docs)
    system_python_result = subprocess.run(
        ["docker", "run", "--rm", "docker-mcp-synaptic", "python", "-c", "import mcp_synaptic"],
        capture_output=True,
        text=True
    )
    assert system_python_result.returncode != 0, "System python should fail - validates troubleshooting docs"
    assert "ModuleNotFoundError" in system_python_result.stderr
    
    # Test that venv python succeeds
    venv_python_result = subprocess.run(
        ["docker", "run", "--rm", "docker-mcp-synaptic", "/app/.venv/bin/python", "-c", "import mcp_synaptic"],
        capture_output=True,
        text=True
    )
    assert venv_python_result.returncode == 0, f"Venv python should succeed: {venv_python_result.stderr}"
    
    # Test specific imports that were failing in production
    import_tests = [
        "from mcp_synaptic.rag.embeddings import EmbeddingManager",
        "from mcp_synaptic.rag.database.core import RAGDatabase",
        "from mcp_synaptic.core.server import SynapticServer",
        "from mcp_synaptic.config.settings import Settings",
    ]
    
    for import_test in import_tests:
        result = subprocess.run(
            ["docker", "run", "--rm", "docker-mcp-synaptic", "/app/.venv/bin/python", "-c", import_test],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Import failed: {import_test}\nError: {result.stderr}"


def test_docker_app_startup():
    """Test that the application can start successfully in Docker (import validation)."""
    # Test that the main module can be imported without errors
    result = subprocess.run(
        ["docker", "run", "--rm", "-e", "EMBEDDING_API_BASE=http://example.com:4000", 
         "docker-mcp-synaptic", "/app/.venv/bin/python", "-c", 
         "from mcp_synaptic import SynapticServer, Settings; print('Startup imports successful')"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, f"App startup imports failed: {result.stderr}"
    assert "Startup imports successful" in result.stdout