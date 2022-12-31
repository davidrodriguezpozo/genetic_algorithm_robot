import pytest
from models import genome


def testGeneToGeneDict():
    spec = genome.Genome.get_gene_spec()
    gene = genome.Genome.get_random_gene(len(spec))
    gene_dict = genome.Genome.get_gene_dict(gene, spec)
    assert "link-recurrence" in gene_dict


def testGenomeToDict():
    spec = genome.Genome.get_gene_spec()
    dna = genome.Genome.get_random_genome(len(spec), 3)
    genome_dicts = genome.Genome.get_genome_dicts(dna, spec)
    assert len(genome_dicts) == 3
