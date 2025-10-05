import { useEffect } from "react";

type RepoAnalProps = {
  repositoryId: number;
};

const RepoAnal = ({ repositoryId }: RepoAnalProps) => {
  useEffect(() => {
    const fetchLatestSession = async () => {
      try {
        const response = await fetch(
          `/api/repositories/${repositoryId}/sessions/latest-with-steps`
        );

        if (!response.ok) {
          throw new Error(`Request failed with status ${response.status}`);
        }

        const data = await response.json();
        console.log("Latest session with steps", data);
      } catch (error) {
        console.error("Failed to fetch latest session with steps", error);
      }
    };

    fetchLatestSession();
  }, [repositoryId]);

  return <div>RepoAnal component</div>;
};

export default RepoAnal;
